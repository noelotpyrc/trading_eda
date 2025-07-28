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
        # Skip parent next() due to position datetime issues
        # super().next()  
        
        # Debug: Show pending orders
        if self.pending_orders:
            print(f"🔧 Processing {len(self.pending_orders)} pending orders")
        
        # Process all pending orders
        for order in self.pending_orders[:]:
            print(f"🔧 Executing order: Type={'BUY' if order.isbuy() else 'SELL'}, Size={order.size}")
            print(f"🔧 Order status before execution: {order.getstatusname()}")
            self._execute_onchain_order(order)
            print(f"🔧 Order status after execution: {order.getstatusname()}")
        
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
                print(f"⚠️ ORDER REJECTED: Invalid price {current_price}")
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
                print(f"⚠️ ORDER REJECTED: Transaction value ${transaction_value:.2f} < minimum ${self.params.min_transaction_size}")
                order.reject()
                return
            
            # Check if we have enough cash/position
            if order.isbuy():
                required_cash = order.size * execution_price
                available_cash = self.get_cash()
                if required_cash > available_cash:
                    print(f"⚠️ ORDER REJECTED: Insufficient cash. Need ${required_cash:,.2f}, have ${available_cash:,.2f}")
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
        
        # Calculate opened/closed amounts for proper Trade tracking
        current_position_size = position.size
        
        if order.isbuy():
            if current_position_size >= 0:
                # Opening long position or adding to long position
                opened = order.size
                closed = 0
                openedvalue = opened * price
                closedvalue = 0.0
                openedcomm = abs(commission)
                closedcomm = 0.0
            else:
                # Reducing short position (buy to cover)
                if abs(order.size) <= abs(current_position_size):
                    # Fully reducing short position
                    opened = 0
                    closed = order.size  # Positive when covering short
                    openedvalue = 0.0
                    closedvalue = closed * price
                    openedcomm = 0.0
                    closedcomm = abs(commission)
                else:
                    # Covering short and opening long
                    closed = abs(current_position_size)
                    opened = order.size - closed
                    closedvalue = closed * price
                    openedvalue = opened * price
                    closedcomm = abs(commission) * (closed / order.size)
                    openedcomm = abs(commission) * (opened / order.size)
        else:  # Sell order
            if current_position_size > 0:
                # Reducing long position
                if abs(order.size) <= current_position_size:
                    # Fully reducing long position
                    closed = abs(order.size)
                    opened = 0
                    closedvalue = closed * price
                    openedvalue = 0.0
                    closedcomm = abs(commission)
                    openedcomm = 0.0
                else:
                    # Closing long and opening short
                    closed = current_position_size
                    opened = abs(order.size) - closed
                    closedvalue = closed * price
                    openedvalue = opened * price
                    closedcomm = abs(commission) * (closed / abs(order.size))
                    openedcomm = abs(commission) * (opened / abs(order.size))
            else:
                # Opening short position or adding to short position
                opened = abs(order.size)
                closed = 0
                openedvalue = opened * price
                closedvalue = 0.0
                openedcomm = abs(commission)
                closedcomm = 0.0

        # Debug: Print execute parameters for TradeAnalyzer troubleshooting
        if hasattr(self, '_debug_trades') and self._debug_trades:
            print(f"🔧 Execute params: size={order.size}, opened={opened}, closed={closed}")
            print(f"   Position before: {position.size}, Position will be: {position.size + order.size}")
        
        # Use parent broker execution for proper Trade tracking
        # The parent broker handles: order.execute(), position.update(), trade creation, notifications
        result = super()._execute(order, ago=0, price=price)
        if hasattr(self, '_debug_trades') and self._debug_trades:
            print(f"🔧 Parent broker execution completed successfully")
        
        # Complete the order
        order.completed()
    
    def getvalue(self, datas=None):
        """Calculate portfolio value as cash + position values"""
        total_value = self.cash
        
        # Add value of all positions
        if datas:
            for data in datas:
                position = self.getposition(data)
                if position.size != 0:
                    current_price = data.close[0] if len(data) > 0 else 0
                    position_value = position.size * current_price
                    total_value += position_value
        
        return total_value


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