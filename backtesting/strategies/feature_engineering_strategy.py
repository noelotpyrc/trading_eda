#!/usr/bin/env python3
"""
Feature Engineering Strategy
Tests WindowFeatureCalculator integration with simple feature-based signals
"""

import sys
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add project root to path for feature engineering import
sys.path.append('/Users/noel/projects/trading_eda')

from solana.feature_engineering.classification_forward.window_feature_calculator import WindowFeatureCalculator


class FeatureEngineeringStrategy(bt.Strategy):
    """
    Strategy that tests WindowFeatureCalculator integration
    Uses simple feature-based trading signals (no ML model yet)
    """
    
    params = (
        # Feature Engineering Parameters  
        ('lookback_windows', [30, 60, 120]),    # Feature calculation windows (seconds)
        ('feature_update_frequency', 1),        # Calculate features every N bars
        
        # Simple Feature-Based Trading Parameters
        ('buy_ratio_threshold', 0.65),          # Buy when buy_ratio_60s > threshold
        ('volume_threshold', 10.0),             # Minimum volume_total_60s
        ('trader_threshold', 3),                # Minimum unique_traders_60s
        
        # Position Management
        ('position_size_pct', 0.2),             # 20% of cash per trade
        ('stop_loss', 0.08),                    # 8% stop loss
        ('take_profit', 0.15),                  # 15% take profit
        ('max_holding_bars', 15),               # Maximum bars to hold position
        
        # Debugging
        ('verbose', True),
        ('log_features', True),                 # Log calculated features
    )
    
    def __init__(self):
        # Raw transaction data from cleaned feed
        self.transaction_price = self.datas[0].transaction_price
        self.sol_amount = self.datas[0].sol_amount
        self.is_buy = self.datas[0].is_buy
        self.trader_id = self.datas[0].trader_id
        self.transaction_size_category = self.datas[0].transaction_size_category
        
        # Feature Engineering Component
        self.feature_calculator = WindowFeatureCalculator()
        
        # Trading State
        self.orders = []
        self.last_features = None
        self.last_feature_update = 0
        self.bars_in_position = 0
        self.entry_price = None
        
        if self.params.verbose:
            print(f"📊 Initialized FeatureEngineeringStrategy")
            print(f"   🔢 Feature windows: {self.params.lookback_windows}s")
            print(f"   🎯 Buy ratio threshold: {self.params.buy_ratio_threshold}")
            print(f"   📈 Position size: {self.params.position_size_pct * 100}%")
            print(f"   🔄 Features calculated every bar (no minimum transaction requirement)")
    
    def next(self):
        """Strategy logic for each data point"""
        current_bar = len(self.data)
        current_date = self.datas[0].datetime.date(0)
        
        # Update bars in position counter
        if self.position.size != 0:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0
        
        # Check for pending orders
        if any(order.status in [order.Submitted, order.Accepted] for order in self.orders):
            return
        
        # Calculate features using real transaction data
        features = self._calculate_features()
        
        # Current market state
        current_price = self.transaction_price[0]
        position_size = self.position.size
        
        # Log current state
        if self.params.verbose and current_bar % 3 == 0:
            feature_str = "No features" if features is None else f"Features OK"
            print(f"📅 {current_date} Bar {current_bar}: Price={current_price:.2f}, "
                  f"Position={position_size:.6f}, {feature_str}")
        
        # Trading logic
        if position_size == 0:  # Not in position
            self._evaluate_entry_signals(features, current_price)
        else:  # In position
            self._evaluate_exit_signals(features, current_price)
    
    def _update_transaction_buffer(self):
        """Update transaction buffer with real transaction data from feed"""
        # No longer needed - we access real transaction data directly from feed
        pass
    
    def _calculate_features(self) -> Optional[Dict]:
        """Calculate features using WindowFeatureCalculator with real transaction data"""
        current_bar = len(self.data)
        
        # Check if we should update features
        if (current_bar - self.last_feature_update) < self.params.feature_update_frequency:
            return self.last_features
        
        try:
            # Get real transaction data from feed
            transactions_df = self.datas[0].df.copy()
            current_timestamp = self.datas[0].datetime.datetime(0)
            
            # Prepare transaction data for WindowFeatureCalculator
            # Convert timezone-aware timestamps to timezone-naive UTC (like debug script)
            if hasattr(transactions_df['block_timestamp'].dtype, 'tz') and transactions_df['block_timestamp'].dt.tz is not None:
                transactions_df['block_timestamp'] = transactions_df['block_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Ensure current_timestamp is timezone-naive UTC
            if hasattr(current_timestamp, 'tz') and current_timestamp.tz is not None:
                current_timestamp = current_timestamp.tz_convert('UTC').replace(tzinfo=None)
            
            # Add required columns for WindowFeatureCalculator
            transactions_df['price'] = 1.0  # Dummy price (not used in volume calculations)
            transactions_df['succeeded'] = True
            transactions_df['mint'] = 'strategy_token'
            
            if self.params.verbose and current_bar <= 5:
                print(f"📊 Real transaction data sample:")
                print(f"   Transactions: {len(transactions_df)}")
                print(f"   Time range: {transactions_df['block_timestamp'].min()} to {transactions_df['block_timestamp'].max()}")
                print(f"   Current timestamp: {current_timestamp}")
                print(f"   Sample data:")
                print(transactions_df[['block_timestamp', 'sol_amount', 'is_buy', 'swapper']].tail(3).to_string())
            
            # Calculate features using WindowFeatureCalculator
            features = self.feature_calculator.calculate_features(
                transactions=transactions_df,
                sample_timestamp=current_timestamp,
                windows=self.params.lookback_windows
            )
            
            # Cache results
            self.last_features = features
            self.last_feature_update = current_bar
            
            # Log features
            if self.params.log_features and features:
                print(f"🔢 Features calculated at bar {current_bar}:")
                print(f"   Total features: {len(features)}")
                
                # Log key trading features
                for window in self.params.lookback_windows:
                    buy_ratio = features.get(f'buy_ratio_{window}s', 0)
                    volume_total = features.get(f'total_volume_{window}s', 0)
                    unique_traders = features.get(f'unique_traders_{window}s', 0)
                    txn_count = features.get(f'total_txns_{window}s', 0)
                    
                    print(f"   {window}s window: BuyRatio={buy_ratio:.3f}, Vol={volume_total:.1f}, "
                          f"Traders={unique_traders}, Txns={txn_count}")
            
            return features
            
        except Exception as e:
            if self.params.verbose:
                print(f"⚠️ Error calculating features: {e}")
                import traceback
                traceback.print_exc()
            # Return default features when calculation fails
            current_timestamp = self.datas[0].datetime.datetime(0)
            return self._get_default_features(current_timestamp)
    
    def _get_default_features(self, timestamp: datetime) -> Dict:
        """Return default feature values when no transaction data is available"""
        features = {}
        
        # Create default features for each window
        for window in self.params.lookback_windows:
            # Volume and transaction features
            features[f'volume_total_{window}s'] = 0.0
            features[f'transaction_count_{window}s'] = 0
            features[f'unique_traders_{window}s'] = 0
            features[f'avg_trade_size_{window}s'] = 0.0
            
            # Buy/sell distribution features
            features[f'buy_ratio_{window}s'] = 0.5  # Neutral
            features[f'buy_volume_{window}s'] = 0.0
            features[f'sell_volume_{window}s'] = 0.0
            
            # Price features
            features[f'price_mean_{window}s'] = 0.0
            features[f'price_trend_{window}s'] = 0.0
            features[f'volatility_{window}s'] = 0.0
            features[f'momentum_{window}s'] = 0.0
            
            # Transaction size distribution
            features[f'small_tx_ratio_{window}s'] = 0.0
            features[f'medium_tx_ratio_{window}s'] = 0.0
            features[f'big_tx_ratio_{window}s'] = 0.0
            features[f'whale_tx_ratio_{window}s'] = 0.0
        
        # Metadata
        features['sample_timestamp'] = timestamp
        features['total_features'] = len(features) - 1  # Exclude metadata
        
        return features
    
    def _evaluate_entry_signals(self, features: Optional[Dict], current_price: float):
        """Evaluate entry signals based on calculated features"""
        if features is None or current_price <= 0:
            return
        
        try:
            # Extract key features for trading decision
            buy_ratio_60s = features.get('buy_ratio_60s', 0.5)
            volume_total_60s = features.get('volume_total_60s', 0)
            unique_traders_60s = features.get('unique_traders_60s', 0)
            
            # Additional context features
            price_trend_60s = features.get('price_trend_60s', 0)
            momentum_60s = features.get('momentum_60s', 0)
            avg_trade_size_60s = features.get('avg_trade_size_{60s}', 0)
            
            # Simple feature-based entry logic
            should_enter = (
                buy_ratio_60s >= self.params.buy_ratio_threshold and
                volume_total_60s >= self.params.volume_threshold and
                unique_traders_60s >= self.params.trader_threshold and
                price_trend_60s > 0  # Positive price trend
            )
            
            if should_enter:
                # Calculate position size
                available_cash = self.broker.get_cash() * self.params.position_size_pct
                
                # Handle large prices with fractional shares
                if current_price > 1e6:  # Large scaled prices
                    shares = available_cash / current_price
                    min_shares = 1e-8
                    if shares >= min_shares:
                        order = self.buy(size=shares)
                        self.orders.append(order)
                        
                        if self.params.verbose:
                            print(f"🛒 FEATURE BUY: {shares:.8f} shares at {current_price:.2f}")
                            print(f"   📊 Signal: BuyRatio={buy_ratio_60s:.3f}, Vol={volume_total_60s:.1f}, "
                                  f"Traders={unique_traders_60s}, Trend={price_trend_60s:.3f}")
                else:
                    # Normal prices with integer shares
                    shares = int(available_cash / current_price)
                    if shares > 0:
                        order = self.buy(size=shares)
                        self.orders.append(order)
                        
                        if self.params.verbose:
                            print(f"🛒 FEATURE BUY: {shares} shares at {current_price:.2f}")
                            print(f"   📊 Signal: BuyRatio={buy_ratio_60s:.3f}, Vol={volume_total_60s:.1f}, "
                                  f"Traders={unique_traders_60s}")
        
        except Exception as e:
            if self.params.verbose:
                print(f"⚠️ Error in entry signal evaluation: {e}")
    
    def _evaluate_exit_signals(self, features: Optional[Dict], current_price: float):
        """Evaluate exit signals based on features and risk management"""
        position_size = self.position.size
        if position_size == 0:
            return
        
        should_exit = False
        exit_reason = ""
        
        # Risk management exits
        if self.entry_price and current_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if pnl_pct <= -self.params.stop_loss:
                should_exit = True
                exit_reason = f"Stop Loss ({pnl_pct*100:.1f}%)"
            elif pnl_pct >= self.params.take_profit:
                should_exit = True
                exit_reason = f"Take Profit ({pnl_pct*100:.1f}%)"
        
        # Maximum holding period
        if self.bars_in_position >= self.params.max_holding_bars:
            should_exit = True
            exit_reason = f"Max Holding ({self.bars_in_position} bars)"
        
        # Feature-based exit signals
        if features:
            try:
                buy_ratio_60s = features.get('buy_ratio_60s', 0.5)
                price_trend_60s = features.get('price_trend_60s', 0)
                momentum_60s = features.get('momentum_60s', 0)
                
                # Exit if market turns bearish
                if (buy_ratio_60s < 0.35 and 
                    price_trend_60s < -0.02 and 
                    momentum_60s < -0.1):
                    should_exit = True
                    exit_reason = f"Bearish Features (BuyRatio={buy_ratio_60s:.3f})"
            
            except Exception as e:
                if self.params.verbose:
                    print(f"⚠️ Error in feature-based exit logic: {e}")
        
        if should_exit:
            order = self.sell(size=abs(position_size))
            self.orders.append(order)
            
            if self.params.verbose:
                print(f"🔔 FEATURE SELL: {abs(position_size):.8f} shares at {current_price:.2f}")
                print(f"   📝 Reason: {exit_reason}")
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                if self.params.verbose:
                    print(f"✅ BUY EXECUTED: {order.executed.size:.8f} shares at {order.executed.price:.2f}")
                    print(f"   💰 Cash remaining: ${self.broker.get_cash():,.2f}")
            else:
                if self.params.verbose:
                    pnl = 0
                    if self.entry_price:
                        pnl = (order.executed.price - self.entry_price) / self.entry_price * 100
                    
                    print(f"✅ SELL EXECUTED: {order.executed.size:.8f} shares at {order.executed.price:.2f}")
                    print(f"   💰 PnL: {pnl:.2f}%, Cash: ${self.broker.get_cash():,.2f}")
                
                # Reset entry price
                self.entry_price = None
                self.bars_in_position = 0
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.verbose:
                print(f"⚠️ ORDER {order.getstatusname()}: Type={'BUY' if order.isbuy() else 'SELL'}")
        
        # Clean up completed/failed orders
        self.orders = [o for o in self.orders if o.status in [order.Submitted, order.Accepted]]
    
    def stop(self):
        """Strategy completion summary"""
        if self.params.verbose:
            final_value = self.broker.get_value()
            print(f'\n📊 === FEATURE ENGINEERING STRATEGY COMPLETED ===')
            print(f'💰 Final Portfolio Value: ${final_value:,.2f}')
            print(f'🔢 Features used: WindowFeatureCalculator with {len(self.params.lookback_windows)} time windows')
            
            if self.last_features:
                print(f'📈 Last calculated features: {len(self.last_features)} total')
                print(f'🎯 Strategy demonstrates: Raw transaction data → Features → Trading signals')