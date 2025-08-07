#!/usr/bin/env python3
"""
Simple Exit Strategy - Hold for N bars then exit
Based on TwoBarExitStrategy from test script
"""

import backtrader as bt

class SimpleExitStrategy(bt.Strategy):
    """
    Strategy that uses execution validation signals for buy/sell decisions
    
    Uses can_execute_buy and can_execute_sell signals from validated data
    instead of raw ML signals, ensuring realistic execution conditions.
    
    Parameters:
    - contrarian_size_sol: Fixed SOL amount for contrarian trades  
    - safe_long_size: Fixed SOL amount for safe long trades
    """
    
    params = dict(
        contrarian_size_sol=1,  # Fixed SOL amount for contrarian trades
        safe_long_size=10,  # Fixed SOL amount for safe long trades
    )

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.strftime('%Y-%m-%d %H:%M:%S'), txt))

    def __init__(self):
        # Keep references to the signal lines (for logging)
        self.safe_long_signal = self.datas[0].safe_long_signal
        self.regime_1_contrarian_signal = self.datas[0].regime_1_contrarian_signal
        
        # Execution validation signals (for actual trading decisions)
        self.can_execute_buy = self.datas[0].can_execute_buy
        self.can_execute_sell = self.datas[0].can_execute_sell
        self.coin_size = self.datas[0].coin_size
        
        # Track entry and orders
        self.entry_bar = None
        self.order_open = None
        self.order_close = None
        
        # Simple persistence
        self.results = []

    def next(self):
        # Record current bar data
        self.results.append({
            'datetime': self.data.datetime.datetime(0),
            'open': self.data.open[0],
            'high': self.data.high[0], 
            'low': self.data.low[0],
            'close': self.data.close[0],
            'volume': self.data.volume[0],
            'safe_long_signal': self.safe_long_signal[0],
            'contrarian_signal': self.regime_1_contrarian_signal[0],
            'can_execute_buy': self.can_execute_buy[0],
            'can_execute_sell': self.can_execute_sell[0],
            'portfolio_value': self.broker.getvalue(),
            'position_size': self.position.size if self.position else 0,
            'action': None,
            'execution_price': None,
            'signal_type': None
        })
        
        # 1) ENTRY logic - use can_execute_buy signal
        if not self.position and self.order_open is None:
            can_buy = self.can_execute_buy[0]
            
            # Skip if price is zero (invalid price data)
            if self.data.close[0] <= 0:
                return
            
            if can_buy == 1:
                # Determine signal type for logging (check which original signal triggered this)
                safe_long = self.safe_long_signal[0]
                contrarian = self.regime_1_contrarian_signal[0]
                signal_type = 'SAFE_LONG' if safe_long == 1 else 'CONTRARIAN' if contrarian == 1 else 'UNKNOWN'
                
                # Position sizing logic (same as original)
                if contrarian == 1:
                    # Regime 1 contrarian: fixed SOL position (parameter)
                    size = self.params.contrarian_size_sol / self.data.close[0]
                else:
                    # Safe long: fixed SOL position (parameter)
                    size = self.params.safe_long_size / self.data.close[0]
                
                self.order_open = self.buy(size=size)
                
                # Mark this bar as having a buy signal
                self.results[-1]['action'] = 'BUY_SIGNAL'
                self.results[-1]['signal_type'] = signal_type
                
                self.log('BUY CREATE, %.2f at bar %d, Type: %s, Size: %.4f (validated)' % (
                    self.data.close[0], len(self), signal_type, size))
        
        # 2) EXIT logic - use can_execute_sell signal
        if self.position and self.entry_bar is not None:
            can_sell = self.can_execute_sell[0]
            
            if can_sell == 1:
                if self.order_close is None:
                    bars_held = len(self) - self.entry_bar
                    self.order_close = self.close()
                    
                    # Mark this bar as having a sell signal
                    self.results[-1]['action'] = 'SELL_SIGNAL'
                    
                    self.log('SELL CREATE, %.2f at bar %d, Held for %d bars (validated)' % (
                        self.data.close[0], len(self), bars_held))

    def notify_order(self, order):
        """Handle order notifications - just for logging and state management"""
        if order.status in (order.Completed, order.Partial):
            if order.isbuy():
                # Record the bar at which the BUY filled
                self.entry_bar = len(self)
                self.order_open = None
                self.log('BUY FILLED at bar %d, price %.2f' % (
                    len(self), order.executed.price))
            elif order.issell():
                # Reset flags ready for the next trade
                self.entry_bar = None
                self.order_close = None
                self.log('SELL FILLED at bar %d, price %.2f' % (
                    len(self), order.executed.price))
        
        # Clean out any rejected / cancelled orders
        elif order.status in (order.Canceled, order.Rejected):
            self.log('Order %s: %s' % (order.getstatusname(), getattr(order, 'info', '')))
            if order.isbuy():
                self.order_open = None
            else:
                self.order_close = None
        

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
        
        pnl = trade.pnl
        pnl_percent = (trade.pnl / abs(trade.value)) * 100 if trade.value else 0
        
        self.log('TRADE CLOSED: PnL %.2f (%.2f%%)' % (pnl, pnl_percent))
    
    def save_results(self, filename='backtest_results.csv'):
        """Save results to CSV with proper signal/execution tracking"""
        import pandas as pd
        
        df = pd.DataFrame(self.results)
        
        # Post-process: Add FILLED actions for the bar after CREATE actions
        for i in range(len(df) - 1):
            current_action = df.loc[i, 'action']
            
            if current_action == 'BUY_SIGNAL':
                # Next bar should be BUY_FILLED with open price as execution price
                df.loc[i + 1, 'action'] = 'BUY_FILLED'
                df.loc[i + 1, 'execution_price'] = df.loc[i + 1, 'open']
                df.loc[i + 1, 'signal_type'] = df.loc[i, 'signal_type']
                
            elif current_action == 'SELL_SIGNAL':
                # Next bar should be SELL_FILLED with open price as execution price
                df.loc[i + 1, 'action'] = 'SELL_FILLED'
                df.loc[i + 1, 'execution_price'] = df.loc[i + 1, 'open']
        
        df.to_csv(filename, index=False)
        print(f"💾 Saved {len(df)} records to {filename}")
        return filename