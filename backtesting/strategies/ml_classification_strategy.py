#!/usr/bin/env python3
"""
ML Classification Strategy
Complete ML trading strategy using WindowFeatureCalculator + ClassificationInference
"""

import sys
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Add project root to path for imports
sys.path.append('/Users/noel/projects/trading_eda')

from solana.feature_engineering.classification_forward.window_feature_calculator import WindowFeatureCalculator
from solana.inference.classification_forward.classification_inference import ClassificationInference


class MLClassificationStrategy(bt.Strategy):
    """
    Complete ML trading strategy that:
    1. Calculates features using WindowFeatureCalculator
    2. Makes predictions using ClassificationInference 
    3. Executes trades based on ML model confidence
    """
    
    params = (
        # Feature Engineering Parameters  
        ('lookback_windows', [30, 60, 120]),    # Feature calculation windows (seconds)
        ('feature_update_frequency', 1),        # Calculate features every N bars
        
        # ML Model Parameters
        ('model_dir', '/Volumes/Extreme SSD/trading_data/solana/models/classification_forward'),
        ('prediction_threshold', 0.6),          # ML confidence threshold for trading
        ('high_confidence_threshold', 0.7),     # High confidence threshold
        ('require_high_confidence', False),     # Only trade on high confidence predictions
        
        # Position Management
        ('position_size_pct', 0.15),             # 15% of cash per trade
        ('stop_loss', None),                    # No stop loss - trust model for 5 bars
        ('take_profit', 0.15),                  # 15% take profit for exceptional moves
        ('max_holding_bars', 5),                # 5 bars = 300s (model prediction horizon)
        
        # Risk Management
        ('max_daily_trades', 50),                # More trades with shorter holds
        ('cooldown_bars', 5),                   # Bars to wait after closing position
        
        # Debugging
        ('verbose', True),
        ('log_features', False),                # Log calculated features
        ('log_predictions', True),              # Log ML predictions
    )
    
    def __init__(self):
        # Raw transaction data from cleaned feed
        self.transaction_price = self.datas[0].transaction_price
        self.sol_amount = self.datas[0].sol_amount
        self.is_buy = self.datas[0].is_buy
        self.trader_id = self.datas[0].trader_id
        self.transaction_size_category = self.datas[0].transaction_size_category
        
        # ML Components
        self.feature_calculator = WindowFeatureCalculator()
        self.ml_inference = ClassificationInference(model_dir=self.params.model_dir)
        
        # Trading State
        self.orders = []
        self.last_features = None
        self.last_feature_update = 0
        self.last_prediction = None
        self.bars_in_position = 0
        self.entry_price = None
        self.cooldown_remaining = 0
        self.daily_trades = 0
        self.last_trade_date = None
        
        # Performance tracking
        self.total_predictions = 0
        self.profitable_signals = 0
        self.high_confidence_signals = 0
        
        if self.params.verbose:
            print(f"🤖 Initialized MLClassificationStrategy")
            print(f"   🔢 Feature windows: {self.params.lookback_windows}s")
            print(f"   🎯 Prediction threshold: {self.params.prediction_threshold}")
            print(f"   📈 Position size: {self.params.position_size_pct * 100}%")
            print(f"   🧠 Model info: {self.ml_inference.get_model_info()}")
    
    def next(self):
        """Strategy logic for each data point"""
        current_bar = len(self.data)
        current_date = self.datas[0].datetime.date(0)
        
        # Reset daily trade counter
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
        
        # Update position tracking
        if self.position.size != 0:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0
        
        # Update cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        
        # Check for pending orders
        if any(order.status in [order.Submitted, order.Accepted] for order in self.orders):
            return
        
        # Calculate features and make ML prediction
        features = self._calculate_features()
        prediction = self._make_ml_prediction(features)
        
        # Current market state
        current_price = self.transaction_price[0]
        position_size = self.position.size
        
        # Log current state
        if self.params.verbose and current_bar % 5 == 0:
            pred_str = f"Score: {prediction['score']:.3f}" if prediction else "No prediction"
            print(f"📅 {current_date} Bar {current_bar}: Price={current_price:.2f}, "
                  f"Position={position_size:.6f}, {pred_str}")
        
        # Trading logic
        if position_size == 0 and self.cooldown_remaining == 0:  # Not in position
            self._evaluate_ml_entry_signals(prediction, current_price)
        elif position_size != 0:  # In position
            self._evaluate_exit_signals(prediction, current_price)
    
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
            if hasattr(transactions_df['block_timestamp'].dtype, 'tz') and transactions_df['block_timestamp'].dt.tz is not None:
                transactions_df['block_timestamp'] = transactions_df['block_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            if hasattr(current_timestamp, 'tz') and current_timestamp.tz is not None:
                current_timestamp = current_timestamp.tz_convert('UTC').replace(tzinfo=None)
            
            # Add required columns for WindowFeatureCalculator
            transactions_df['price'] = 1.0
            transactions_df['succeeded'] = True
            transactions_df['mint'] = 'ml_strategy_token'
            
            # Calculate features
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
                for window in self.params.lookback_windows:
                    buy_ratio = features.get(f'buy_ratio_{window}s', 0)
                    volume_total = features.get(f'total_volume_{window}s', 0)
                    unique_traders = features.get(f'unique_traders_{window}s', 0)
                    print(f"   {window}s: BuyRatio={buy_ratio:.3f}, Vol={volume_total:.1f}, Traders={unique_traders}")
            
            return features
            
        except Exception as e:
            if self.params.verbose:
                print(f"⚠️ Error calculating features: {e}")
            return None
    
    def _make_ml_prediction(self, features: Optional[Dict]) -> Optional[Dict]:
        """Make ML prediction using ClassificationInference"""
        if features is None:
            return None
        
        try:
            # Convert features to DataFrame for ML model
            # Remove metadata fields that aren't model features
            feature_data = {k: v for k, v in features.items() 
                          if k not in ['sample_timestamp', 'total_transactions', 'total_features']}
            
            if not feature_data:
                return None
            
            # Create DataFrame with single row for prediction
            features_df = pd.DataFrame([feature_data])
            
            # Get ML prediction
            prediction_result = self.ml_inference.predict_full_output(
                features_df, 
                threshold=self.params.prediction_threshold
            )
            
            # Extract single prediction
            prediction = {
                'score': float(prediction_result['scores'][0]),  # Probability of profitable trade
                'label': int(prediction_result['labels'][0]),   # 0=unprofitable, 1=profitable
                'confidence': float(prediction_result['confidence'][0]),  # Max probability
                'high_confidence': bool(prediction_result['high_confidence_mask'][0]),
                'threshold_used': prediction_result['threshold_used'],
                'n_features': len(feature_data)
            }
            
            # Cache prediction
            self.last_prediction = prediction
            self.total_predictions += 1
            
            if prediction['high_confidence']:
                self.high_confidence_signals += 1
            
            # Log prediction
            if self.params.log_predictions:
                print(f"🧠 ML Prediction: Score={prediction['score']:.3f}, "
                      f"Label={'BUY' if prediction['label'] == 1 else 'HOLD'}, "
                      f"Confidence={prediction['confidence']:.3f}, "
                      f"HighConf={'✅' if prediction['high_confidence'] else '❌'}")
            
            return prediction
            
        except Exception as e:
            if self.params.verbose:
                print(f"⚠️ Error making ML prediction: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _evaluate_ml_entry_signals(self, prediction: Optional[Dict], current_price: float):
        """Evaluate ML-based entry signals"""
        if (prediction is None or 
            current_price <= 0 or 
            self.daily_trades >= self.params.max_daily_trades):
            return
        
        try:
            # ML entry conditions
            ml_signal = prediction['label'] == 1  # Model predicts profitable
            confidence_ok = (not self.params.require_high_confidence or 
                           prediction['high_confidence'])
            score_ok = prediction['score'] >= self.params.prediction_threshold
            
            should_enter = ml_signal and confidence_ok and score_ok
            
            if should_enter:
                # Calculate position size
                available_cash = self.broker.get_cash() * self.params.position_size_pct
                
                # Handle large scaled prices with fractional shares
                if current_price > 1e6:
                    shares = available_cash / current_price
                    min_shares = 1e-8
                    if shares >= min_shares:
                        order = self.buy(size=shares)
                        self.orders.append(order)
                        self.daily_trades += 1
                        
                        if self.params.verbose:
                            print(f"🤖 ML BUY: {shares:.8f} shares at {current_price:.2f}")
                            print(f"   📊 Signal: Score={prediction['score']:.3f}, "
                                  f"Confidence={prediction['confidence']:.3f}, "
                                  f"HighConf={'✅' if prediction['high_confidence'] else '❌'}")
                else:
                    # Normal prices with integer shares
                    shares = int(available_cash / current_price)
                    if shares > 0:
                        order = self.buy(size=shares)
                        self.orders.append(order)
                        self.daily_trades += 1
                        
                        if self.params.verbose:
                            print(f"🤖 ML BUY: {shares} shares at {current_price:.2f}")
                            print(f"   📊 Signal: Score={prediction['score']:.3f}, "
                                  f"Confidence={prediction['confidence']:.3f}")
        
        except Exception as e:
            if self.params.verbose:
                print(f"⚠️ Error in ML entry signal evaluation: {e}")
    
    def _evaluate_exit_signals(self, prediction: Optional[Dict], current_price: float):
        """Evaluate exit signals based on ML predictions and risk management"""
        position_size = self.position.size
        if position_size == 0:
            return
        
        should_exit = False
        exit_reason = ""
        
        # Risk management exits
        if self.entry_price and current_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Only take profit exit - no stop loss (trust model for 5 bars)
            if self.params.take_profit and pnl_pct >= self.params.take_profit:
                should_exit = True
                exit_reason = f"Take Profit ({pnl_pct*100:.1f}%)"
                self.profitable_signals += 1
        
        # Maximum holding period
        if self.bars_in_position >= self.params.max_holding_bars:
            should_exit = True
            exit_reason = f"Max Holding ({self.bars_in_position} bars)"
        
        # ML-based exit signals
        if prediction and not should_exit:
            try:
                # Exit if model confidence turns bearish
                if (prediction['label'] == 0 and  # Model predicts unprofitable
                    prediction['confidence'] > 0.7 and  # High confidence in bearish signal
                    prediction['score'] < 0.4):  # Low probability of profit
                    should_exit = True
                    exit_reason = f"ML Bearish Signal (Score={prediction['score']:.3f})"
            
            except Exception as e:
                if self.params.verbose:
                    print(f"⚠️ Error in ML exit logic: {e}")
        
        if should_exit:
            order = self.sell(size=abs(position_size))
            self.orders.append(order)
            self.cooldown_remaining = self.params.cooldown_bars
            
            if self.params.verbose:
                print(f"🔔 ML SELL: {abs(position_size):.8f} shares at {current_price:.2f}")
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
            print(f'\n🤖 === ML CLASSIFICATION STRATEGY COMPLETED ===')
            print(f'💰 Final Portfolio Value: ${final_value:,.2f}')
            print(f'🧠 Total ML Predictions: {self.total_predictions}')
            print(f'📈 Profitable Signals: {self.profitable_signals}')
            print(f'🎯 High Confidence Signals: {self.high_confidence_signals}')
            print(f'📊 Daily Trades: {self.daily_trades}')
            
            if self.total_predictions > 0:
                profit_rate = self.profitable_signals / self.total_predictions * 100
                high_conf_rate = self.high_confidence_signals / self.total_predictions * 100
                print(f'📈 Profit Signal Rate: {profit_rate:.1f}%')
                print(f'🎯 High Confidence Rate: {high_conf_rate:.1f}%')
            
            print(f'🔧 Strategy: WindowFeatureCalculator + ClassificationInference')
            print(f'🎯 ML Pipeline: Raw data → Features → Predictions → Trading signals')