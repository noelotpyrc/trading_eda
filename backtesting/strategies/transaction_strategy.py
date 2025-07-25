#!/usr/bin/env python3
"""
Solana Transaction Strategy
Example strategy implementation for trading based on transaction-level signals

This is a reference implementation showing how to use SolanaTransactionFeed
for onchain transaction-based trading strategies.
"""

import backtrader as bt


class SolanaTransactionStrategy(bt.Strategy):
    """
    Strategy that works directly with transaction-level data
    
    Uses buy pressure, volume momentum, and whale activity signals
    to make trading decisions on onchain transaction data.
    """
    
    params = (
        ('buy_pressure_threshold', 0.6),    # Buy when buy pressure > threshold
        ('volume_momentum_threshold', 0.1), # Require positive volume momentum
        ('whale_ratio_threshold', 0.002),   # Whale activity threshold
        ('min_transaction_count', 10),      # Minimum transactions per window
        ('position_size_pct', 0.9),         # Position size as % of cash
        ('stop_loss', 0.05),
        ('take_profit', 0.15),
        ('verbose', True),
    )
    
    def __init__(self):
        # Reference to transaction data lines
        self.transaction_price = self.datas[0].transaction_price
        self.sol_amount = self.datas[0].sol_amount
        self.is_buy = self.datas[0].is_buy
        self.trader_id = self.datas[0].trader_id
        self.transaction_size_category = self.datas[0].transaction_size_category
        self.cumulative_volume = self.datas[0].cumulative_volume
        self.transaction_count = self.datas[0].transaction_count
        self.buy_pressure = self.datas[0].buy_pressure
        self.volume_momentum = self.datas[0].volume_momentum
        
        # Track orders and performance
        self.order = None
        self.buy_price = None
        self.trade_count = 0
        
        if self.params.verbose:
            print(f"✅ Initialized SolanaTransactionStrategy")
    
    def next(self):
        """Strategy logic for each data point"""
        current_date = self.datas[0].datetime.date(0)
        
        # Check if we have a pending order
        if self.order:
            return
        
        # Get current signals
        buy_pressure = self.buy_pressure[0]
        volume_momentum = self.volume_momentum[0]
        whale_ratio = self.transaction_size_category[0]  # Using as whale ratio
        txn_count = self.transaction_count[0]
        current_price = self.transaction_price[0]
        
        # Current position
        position_size = self.position.size
        
        # Log current state
        if self.params.verbose and len(self.datas[0]) % 50 == 0:
            print(f'Date: {current_date}, Price: {current_price:.4f}, '
                  f'Buy Pressure: {buy_pressure:.3f}, Volume Mom: {volume_momentum:.3f}, '
                  f'Txns: {txn_count}, Position: {position_size}, Cash: ${self.broker.get_cash():.2f}')
        
        # Trading logic
        if not position_size:  # Not in position
            # Buy conditions
            should_buy = (
                buy_pressure >= self.params.buy_pressure_threshold and
                volume_momentum >= self.params.volume_momentum_threshold and
                txn_count >= self.params.min_transaction_count
            )
            
            # Enhanced buy signal during whale activity
            if whale_ratio > self.params.whale_ratio_threshold:
                should_buy = should_buy or (buy_pressure >= 0.55 and txn_count >= 5)
            
            if should_buy:
                # Calculate position size
                available_cash = self.broker.get_cash() * self.params.position_size_pct
                shares = int(available_cash / current_price)
                
                if shares > 0:
                    self.order = self.buy(size=shares)
                    if self.params.verbose:
                        print(f'BUY ORDER: {shares} shares at {current_price:.4f}, '
                              f'Buy Pressure: {buy_pressure:.3f}, Whale: {whale_ratio:.4f}')
        
        else:  # In position
            # Sell conditions
            if self.buy_price:
                pnl_pct = (current_price - self.buy_price) / self.buy_price
                
                should_sell = False
                sell_reason = ""
                
                # Stop loss
                if pnl_pct <= -self.params.stop_loss:
                    should_sell = True
                    sell_reason = f"Stop Loss ({pnl_pct*100:.1f}%)"
                
                # Take profit
                elif pnl_pct >= self.params.take_profit:
                    should_sell = True
                    sell_reason = f"Take Profit ({pnl_pct*100:.1f}%)"
                
                # Signal-based sell
                elif (buy_pressure < 0.4 and volume_momentum < -0.1):
                    should_sell = True
                    sell_reason = f"Signal Sell (Pressure: {buy_pressure:.3f})"
                
                if should_sell:
                    self.order = self.sell(size=position_size)
                    if self.params.verbose:
                        print(f'SELL ORDER: {position_size} shares at {current_price:.4f}, '
                              f'Reason: {sell_reason}')
                        print(f'   Order created: {self.order is not None}')
                        print(f'   Order size: {self.order.size if self.order else "None"}')
    
    def notify_order(self, order):
        """Handle order notifications"""
        if self.params.verbose:
            print(f'📋 ORDER NOTIFICATION: Status={order.status}, Type={"BUY" if order.isbuy() else "SELL"}')
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                if self.params.verbose:
                    print(f'✅ BUY EXECUTED: {order.executed.size} shares at {order.executed.price:.4f}')
                    print(f'   Cash remaining: ${self.broker.get_cash():.2f}')
                    print(f'   Position size: {self.position.size}')
            else:
                if self.params.verbose:
                    print(f'✅ SELL EXECUTED: {order.executed.size} shares at {order.executed.price:.4f}')
                    print(f'   Cash after sell: ${self.broker.get_cash():.2f}')
                self.trade_count += 1
                self.buy_price = None  # Reset for next trade
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.verbose:
                print(f'⚠️ ORDER {order.status}: Type={"BUY" if order.isbuy() else "SELL"}')
        
        self.order = None
    
    def stop(self):
        """Strategy completion summary"""
        if self.params.verbose:
            final_value = self.broker.get_value()
            print(f'\n=== TRANSACTION STRATEGY COMPLETED ===')
            print(f'Final Portfolio Value: ${final_value:,.2f}')
            print(f'Total Trades: {self.trade_count}')


class DiagnosticSolanaStrategy(SolanaTransactionStrategy):
    """
    Enhanced version with comprehensive diagnostic logging
    
    Extends SolanaTransactionStrategy with detailed market state tracking,
    signal analysis, and execution diagnostics for strategy debugging.
    """
    
    def __init__(self):
        super().__init__()
        self.detailed_logs = []
        print("🔍 Initialized DiagnosticSolanaStrategy with enhanced logging")
    
    def next(self):
        """Enhanced strategy logic with detailed logging"""
        current_date = self.datas[0].datetime.date(0)
        
        # Log market state every period
        self.log_market_state()
        
        # Call parent strategy logic
        super().next()
    
    def log_market_state(self):
        """Log comprehensive market state"""
        market_state = {
            'datetime': self.datas[0].datetime.date(0),
            'price': self.transaction_price[0],
            'volume': self.sol_amount[0],
            'buy_pressure': self.buy_pressure[0],
            'volume_momentum': self.volume_momentum[0],
            'whale_ratio': self.transaction_size_category[0],
            'transaction_count': self.transaction_count[0],
            'position_size': self.position.size,
            'cash': self.broker.get_cash(),
            'portfolio_value': self.broker.get_value()
        }
        
        self.detailed_logs.append(market_state)
        
        # Detailed logging every 10 periods
        if len(self.datas[0]) % 10 == 0:
            print(f"\n🔍 MARKET DIAGNOSIS - {market_state['datetime']}")
            print(f"   💰 Price: {market_state['price']:.4f}")
            print(f"   📈 Buy Pressure: {market_state['buy_pressure']:.3f}")
            print(f"   📊 Volume Momentum: {market_state['volume_momentum']:.3f}")
            print(f"   🐋 Whale Activity: {market_state['whale_ratio']:.4f}")
            print(f"   📋 Transactions: {market_state['transaction_count']}")
            print(f"   🎯 Position: {market_state['position_size']}")
            print(f"   💵 Cash: ${market_state['cash']:,.2f}")
            print(f"   📊 Portfolio: ${market_state['portfolio_value']:,.2f}")
    
    def notify_order(self, order):
        """Enhanced order notifications with execution details"""
        super().notify_order(order)
        
        if order.status in [order.Completed]:
            execution_details = {
                'datetime': self.datas[0].datetime.date(0),
                'type': 'BUY' if order.isbuy() else 'SELL',
                'size': order.executed.size,
                'price': order.executed.price,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'buy_pressure': self.buy_pressure[0],
                'volume_momentum': self.volume_momentum[0],
                'whale_ratio': self.transaction_size_category[0],
            }
            
            print(f"\n🎯 EXECUTION ANALYSIS")
            print(f"   📅 Date: {execution_details['datetime']}")
            print(f"   📋 Type: {execution_details['type']}")
            print(f"   📊 Size: {execution_details['size']}")
            print(f"   💰 Price: {execution_details['price']:.4f}")
            print(f"   💵 Value: ${execution_details['value']:,.2f}")
            print(f"   🔧 Commission: ${execution_details['commission']:.2f}")
            print(f"   📈 Market Conditions:")
            print(f"      - Buy Pressure: {execution_details['buy_pressure']:.3f}")
            print(f"      - Volume Momentum: {execution_details['volume_momentum']:.3f}")
            print(f"      - Whale Activity: {execution_details['whale_ratio']:.4f}")
    
    def stop(self):
        """Enhanced completion summary with detailed analysis"""
        super().stop()
        
        print(f"\n📊 DIAGNOSTIC SUMMARY")
        print(f"   Total Market States Logged: {len(self.detailed_logs)}")
        
        if self.detailed_logs:
            import pandas as pd
            df = pd.DataFrame(self.detailed_logs)
            
            print(f"   📈 Price Range: ${df['price'].min():.4f} - ${df['price'].max():.4f}")
            print(f"   📊 Avg Buy Pressure: {df['buy_pressure'].mean():.3f}")
            print(f"   📈 Avg Volume Momentum: {df['volume_momentum'].mean():.3f}")
            print(f"   🐋 Max Whale Activity: {df['whale_ratio'].max():.4f}")