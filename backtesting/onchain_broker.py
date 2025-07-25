#!/usr/bin/env python3
"""
Onchain Transaction Broker for Backtrader
Handles order execution for onchain transaction-based trading
"""

import backtrader as bt
from typing import Optional, Dict, Any


class OnchainBroker(bt.brokers.BackBroker):
    """
    Custom broker for onchain transaction execution
    
    Unlike traditional brokers, onchain execution:
    - Uses actual transaction prices directly
    - Has minimal slippage (just gas costs)
    - Executes immediately at current transaction price
    - No bid/ask spreads or market depth considerations
    """
    
    params = (
        ('gas_cost_pct', 0.001),      # Gas cost as percentage (0.1%)
        ('slippage_pct', 0.0005),     # Minimal slippage (0.05%)
        ('min_transaction_size', 0.01), # Minimum transaction size in SOL
    )
    
    def __init__(self):
        super().__init__()
        self.pending_orders = []
        
    def submit(self, order, check=True):
        """Submit order for onchain execution"""
        if order.exectype != bt.Order.Market:
            # Only support market orders for onchain execution
            order.reject()
            return order
        
        # Add to pending orders for immediate processing
        self.pending_orders.append(order)
        order.submit()
        return order
    
    def next(self):
        """Process pending orders using current transaction data"""
        super().next()
        
        # Debug: Show pending orders
        if self.pending_orders:
            print(f"🔧 Processing {len(self.pending_orders)} pending orders")
        
        # Process all pending orders
        for order in self.pending_orders[:]:
            print(f"🔧 Executing order: Type={'BUY' if order.isbuy() else 'SELL'}, Size={order.size}")
            self._execute_onchain_order(order)
        
        self.pending_orders.clear()
    
    def _execute_onchain_order(self, order):
        """Execute order using onchain transaction mechanics"""
        try:
            data = order.data
            
            # Get current transaction price
            if hasattr(data, 'transaction_price'):
                # Use transaction-specific price
                current_price = data.transaction_price[0]
            else:
                # Fallback to close price
                current_price = data.close[0]
            
            if current_price <= 0:
                order.reject()
                return
            
            # Apply onchain execution costs
            if order.isbuy():
                # For buys, we pay gas + slippage
                execution_price = current_price * (1 + self.params.gas_cost_pct + self.params.slippage_pct)
            else:
                # For sells, we receive less due to gas costs
                execution_price = current_price * (1 - self.params.gas_cost_pct - self.params.slippage_pct)
            
            # Check minimum transaction size
            transaction_value = abs(order.size) * execution_price
            if transaction_value < self.params.min_transaction_size:
                order.reject()
                return
            
            # Check if we have enough cash/position
            if order.isbuy():
                required_cash = order.size * execution_price
                if required_cash > self.get_cash():
                    order.reject()
                    return
            else:
                # Check if we have enough position to sell
                position = self.getposition(order.data)
                required_size = abs(order.size)  # order.size is negative for sells
                if required_size > position.size:
                    print(f"⚠️ Insufficient position: need {required_size}, have {position.size}")
                    order.reject()
                    return
            
            # Execute the order
            self._execute_order(order, execution_price)
            
        except Exception as e:
            print(f"⚠️ Onchain execution error: {e}")
            order.reject()
    
    def _execute_order(self, order, price):
        """Execute order at specified price using parent broker execution"""
        # Get current position
        position = self.getposition(order.data)
        
        # Calculate commission (gas costs)
        commission = order.size * price * self.params.gas_cost_pct
        
        # Use parent broker's execution mechanism
        # Execute with corrected parameters for Backtrader
        order.execute(
            order.data.datetime.datetime(0),  # dt
            order.size,                       # size  
            price,                           # price
            order.size,                      # closed
            order.size * price,              # closedvalue
            commission,                      # closedcomm
            0,                              # opened
            0.0,                            # openedvalue
            0.0,                            # openedcomm
            0.0,                            # margin
            0.0,                            # pnl
            position.size,                   # psize (current position size)
            position.price                   # pprice (current position price)
        )
        
        # Update position and cash manually
        if order.isbuy():
            self.cash -= (order.size * price + commission)
            position.update(order.size, price)
        else:
            self.cash += (order.size * price - commission)  
            position.update(-order.size, price)
        
        # Complete the order
        order.completed()
        
        # Notify the strategy
        if hasattr(order.owner, 'notify_order'):
            order.owner.notify_order(order)


class OnchainCommissionInfo(bt.CommInfoBase):
    """
    Commission structure for onchain transactions
    Models gas costs and network fees
    """
    
    params = (
        ('commission', 0.001),        # 0.1% gas cost
        ('mult', 1.0),                # Multiplier
        ('margin', None),             # No margin trading
        ('commtype', bt.CommInfoBase.COMM_PERC),  # Percentage-based
        ('stocklike', True),          # Stock-like behavior
        ('percabs', False),           # Percentage of transaction value
    )
    
    def getcommission(self, size, price):
        """Calculate gas costs for transaction"""
        return abs(size) * price * self.params.commission


def setup_onchain_broker(cerebro, initial_cash=100000):
    """
    Setup cerebro with onchain broker and commission structure
    
    Args:
        cerebro: Backtrader cerebro engine
        initial_cash: Starting cash amount
    """
    # Set custom onchain broker
    onchain_broker = OnchainBroker()
    onchain_broker.set_cash(initial_cash)
    
    # Set onchain commission structure
    onchain_commission = OnchainCommissionInfo()
    onchain_broker.addcommissioninfo(onchain_commission)
    
    # Apply to cerebro
    cerebro.setbroker(onchain_broker)
    
    print(f"✅ Onchain broker configured:")
    print(f"   Initial cash: ${initial_cash:,.2f}")
    print(f"   Gas cost: {onchain_broker.params.gas_cost_pct * 100:.3f}%")
    print(f"   Slippage: {onchain_broker.params.slippage_pct * 100:.3f}%")
    
    return onchain_broker