#!/usr/bin/env python3
"""
Simple Transaction Strategy
Clean strategy that works with the cleaned SolanaTransactionFeed
Demonstrates proper separation of concerns - data feed provides raw data, strategy handles intelligence
"""

import backtrader as bt
import numpy as np
from collections import deque


class SimpleTransactionStrategy(bt.Strategy):
    """
    Simple strategy that works with cleaned transaction data feed
    
    Features:
    - Uses only raw transaction data from cleaned SolanaTransactionFeed
    - Calculates its own simple indicators internally
    - Clean separation: data feed = raw data, strategy = intelligence
    - Suitable for examples and educational purposes
    """
    
    params = (
        ('buy_ratio_threshold', 0.6),       # Buy when buy ratio > threshold
        ('volume_threshold', 5.0),          # Minimum SOL volume per window
        ('trader_diversity_threshold', 3),   # Minimum unique traders
        ('position_size_pct', 0.8),         # Position size as % of cash
        ('stop_loss', 0.05),                # 5% stop loss
        ('take_profit', 0.15),              # 15% take profit
        ('lookback_periods', 3),            # Periods to analyze for momentum
        ('verbose', True),
    )
    
    def __init__(self):
        # Raw transaction data from cleaned feed
        self.transaction_price = self.datas[0].transaction_price
        self.sol_amount = self.datas[0].sol_amount
        self.is_buy = self.datas[0].is_buy
        self.trader_id = self.datas[0].trader_id
        self.transaction_size_category = self.datas[0].transaction_size_category
        
        # Strategy-specific tracking
        self.order = None
        self.buy_price = None
        self.trade_count = 0
        
        # Rolling data for momentum calculation
        self.volume_history = deque(maxlen=self.params.lookback_periods)
        self.buy_ratio_history = deque(maxlen=self.params.lookback_periods)
        
        if self.params.verbose:
            print(f"✅ Initialized SimpleTransactionStrategy")
            print(f"   📊 Parameters: buy_ratio={self.params.buy_ratio_threshold}, "
                  f"volume={self.params.volume_threshold}, traders={self.params.trader_diversity_threshold}")
    
    def next(self):
        """Strategy logic for each data point"""
        current_date = self.datas[0].datetime.date(0)
        
        # Check if we have a pending order
        if self.order:
            return
        
        # Get raw transaction data
        current_price = self.transaction_price[0]
        volume = self.sol_amount[0]
        buy_ratio = self.is_buy[0]
        unique_traders = self.trader_id[0]
        avg_size_category = self.transaction_size_category[0]
        
        # Update rolling history
        self.volume_history.append(volume)
        self.buy_ratio_history.append(buy_ratio)
        
        # Calculate simple momentum indicators
        volume_momentum = self._calculate_volume_momentum()
        buy_pressure_trend = self._calculate_buy_pressure_trend()
        
        # Current position
        position_size = self.position.size
        
        # Log current state (reduced frequency)
        if self.params.verbose and len(self.datas[0]) % 10 == 0:
            print(f'📅 {current_date}: Price={current_price:.4f}, Vol={volume:.2f}, '
                  f'BuyRatio={buy_ratio:.3f}, Traders={unique_traders}, Pos={position_size}')
        
        # Trading logic
        if not position_size:  # Not in position
            # Simple buy conditions based on raw data
            should_buy = (
                buy_ratio >= self.params.buy_ratio_threshold and
                volume >= self.params.volume_threshold and
                unique_traders >= self.params.trader_diversity_threshold and
                volume_momentum > 0  # Positive volume momentum
            )
            
            # Enhanced signal during whale activity (high average size category)
            if avg_size_category > 1.5:  # Above medium size
                should_buy = should_buy or (
                    buy_ratio >= 0.55 and 
                    volume >= self.params.volume_threshold * 0.5
                )
            
            if should_buy and current_price > 0:
                # Calculate position size with fractional shares for large price scales
                available_cash = self.broker.get_cash() * self.params.position_size_pct
                shares = available_cash / current_price
                
                # Use fractional shares if price is very large, otherwise use integer shares
                if current_price > 1e6:  # For scaled prices > 1 million
                    min_shares = 0.000001  # Minimum fractional position
                    if shares >= min_shares:
                        self.order = self.buy(size=shares)
                        if self.params.verbose:
                            print(f'🛒 BUY: {shares:.6f} shares at {current_price:.4f}, '
                                  f'Vol={volume:.2f}, BuyRatio={buy_ratio:.3f}, '
                                  f'VolMomentum={volume_momentum:.3f}')
                else:
                    # Traditional integer shares for normal prices
                    shares = int(shares)
                    if shares > 0:
                        self.order = self.buy(size=shares)
                        if self.params.verbose:
                            print(f'🛒 BUY: {shares} shares at {current_price:.4f}, '
                                  f'Vol={volume:.2f}, BuyRatio={buy_ratio:.3f}, '
                                  f'VolMomentum={volume_momentum:.3f}')
        
        else:  # In position
            # Sell conditions
            if self.buy_price and current_price > 0:
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
                
                # Signal-based sell (negative momentum + low buy pressure)
                elif (buy_ratio < 0.4 and 
                      volume_momentum < -0.1 and 
                      buy_pressure_trend < 0):
                    should_sell = True
                    sell_reason = f"Negative Momentum (BuyRatio={buy_ratio:.3f})"
                
                if should_sell:
                    self.order = self.sell(size=position_size)
                    if self.params.verbose:
                        print(f'🔔 SELL: {position_size} shares at {current_price:.4f}, '
                              f'Reason: {sell_reason}')
    
    def _calculate_volume_momentum(self):
        """Calculate simple volume momentum"""
        if len(self.volume_history) < 2:
            return 0.0
        
        # Compare recent vs earlier volumes
        if len(self.volume_history) >= 3:
            recent_avg = np.mean(list(self.volume_history)[-2:])
            earlier_avg = np.mean(list(self.volume_history)[:-2])
            
            if earlier_avg > 0:
                momentum = (recent_avg - earlier_avg) / earlier_avg
                return np.clip(momentum, -1.0, 1.0)
        
        return 0.0
    
    def _calculate_buy_pressure_trend(self):
        """Calculate buy pressure trend"""
        if len(self.buy_ratio_history) < 2:
            return 0.0
        
        # Simple trend: current vs previous
        current = self.buy_ratio_history[-1]
        previous = self.buy_ratio_history[-2] if len(self.buy_ratio_history) >= 2 else 0.5
        
        return current - previous
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                if self.params.verbose:
                    print(f'✅ BUY EXECUTED: {order.executed.size} shares at {order.executed.price:.4f}')
                    print(f'   💰 Cash remaining: ${self.broker.get_cash():.2f}')
            else:
                if self.params.verbose:
                    # Calculate actual PnL for debugging
                    position_value = abs(order.executed.size) * order.executed.price
                    buy_value = abs(order.executed.size) * self.buy_price if self.buy_price else 0
                    raw_pnl = position_value - buy_value
                    
                    print(f'✅ SELL EXECUTED: {order.executed.size} shares at {order.executed.price:.4f}')
                    print(f'   📊 Position value: ${position_value:,.2f}')
                    print(f'   📊 Buy value: ${buy_value:,.2f}') 
                    print(f'   📊 Raw PnL: ${raw_pnl:,.2f}')
                    print(f'   💰 Cash after sell: ${self.broker.get_cash():.2f}')
                    print(f'   💼 Portfolio value: ${self.broker.get_value():.2f}')
                self.trade_count += 1
                self.buy_price = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.verbose:
                print(f'⚠️ ORDER {order.status}: Type={"BUY" if order.isbuy() else "SELL"}')
        
        self.order = None
    
    def stop(self):
        """Strategy completion summary"""
        if self.params.verbose:
            final_value = self.broker.get_value()
            print(f'\n📊 === SIMPLE TRANSACTION STRATEGY COMPLETED ===')
            print(f'💰 Final Portfolio Value: ${final_value:,.2f}')
            print(f'🔄 Total Trades: {self.trade_count}')
            print(f'📈 Strategy used only raw transaction data + simple internal calculations')


class DiagnosticSimpleStrategy(SimpleTransactionStrategy):
    """
    Enhanced version with comprehensive diagnostic logging
    Shows how raw transaction data can be used for detailed market analysis
    """
    
    def __init__(self):
        super().__init__()
        self.detailed_logs = []
        print("🔍 Initialized DiagnosticSimpleStrategy with enhanced logging")
    
    def next(self):
        """Enhanced strategy logic with detailed logging"""
        current_date = self.datas[0].datetime.date(0)
        
        # Log market state every period
        self.log_market_state()
        
        # Call parent strategy logic
        super().next()
    
    def log_market_state(self):
        """Log comprehensive market state using only raw data"""
        current_price = self.transaction_price[0]
        volume = self.sol_amount[0]
        buy_ratio = self.is_buy[0]
        unique_traders = self.trader_id[0]
        avg_size_category = self.transaction_size_category[0]
        
        # Calculate derived metrics
        volume_momentum = self._calculate_volume_momentum()
        buy_pressure_trend = self._calculate_buy_pressure_trend()
        
        market_state = {
            'datetime': self.datas[0].datetime.date(0),
            'price': current_price,
            'volume': volume,
            'buy_ratio': buy_ratio,
            'unique_traders': unique_traders,
            'avg_size_category': avg_size_category,
            'volume_momentum': volume_momentum,
            'buy_pressure_trend': buy_pressure_trend,
            'position_size': self.position.size,
            'cash': self.broker.get_cash(),
            'portfolio_value': self.broker.get_value()
        }
        
        self.detailed_logs.append(market_state)
        
        # Detailed logging every 5 periods
        if len(self.datas[0]) % 5 == 0:
            print(f"\n🔍 MARKET DIAGNOSIS - {market_state['datetime']}")
            print(f"   💰 Price: {market_state['price']:.4f}")
            print(f"   📊 Volume: {market_state['volume']:.2f} SOL")
            print(f"   📈 Buy Ratio: {market_state['buy_ratio']:.3f}")
            print(f"   👥 Traders: {market_state['unique_traders']}")
            print(f"   📏 Avg Size Category: {market_state['avg_size_category']:.2f}")
            print(f"   🚀 Volume Momentum: {market_state['volume_momentum']:.3f}")
            print(f"   📊 Buy Pressure Trend: {market_state['buy_pressure_trend']:.3f}")
            print(f"   🎯 Position: {market_state['position_size']}")
            print(f"   💵 Cash: ${market_state['cash']:,.2f}")
    
    def stop(self):
        """Enhanced completion summary with analysis"""
        super().stop()
        
        print(f"\n📊 DIAGNOSTIC SUMMARY")
        print(f"   📈 Total Market States Logged: {len(self.detailed_logs)}")
        
        if self.detailed_logs:
            import pandas as pd
            df = pd.DataFrame(self.detailed_logs)
            
            print(f"   💰 Price Range: ${df['price'].min():.4f} - ${df['price'].max():.4f}")
            print(f"   📊 Avg Volume: {df['volume'].mean():.2f} SOL")
            print(f"   📈 Avg Buy Ratio: {df['buy_ratio'].mean():.3f}")
            print(f"   👥 Avg Traders: {df['unique_traders'].mean():.1f}")
            print(f"   🚀 Max Volume Momentum: {df['volume_momentum'].max():.3f}")
            print(f"   📊 Strategy demonstrates clean architecture: raw data + internal intelligence")