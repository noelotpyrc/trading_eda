#!/usr/bin/env python3
"""
Solana Transaction-Level Data Feed
Direct processing of individual Solana transactions without OHLCV aggregation
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os

# Add the feature engineering path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'solana', 'feature_engineering', 'classification_forward'))

try:
    from window_feature_calculator import WindowFeatureCalculator
except ImportError:
    print("Warning: WindowFeatureCalculator not found. Feature calculation will be disabled.")
    WindowFeatureCalculator = None


class SolanaTransactionFeed(bt.feeds.DataBase):
    """
    Custom data feed for processing individual Solana transactions
    Bypasses OHLCV aggregation to work with raw transaction data
    """
    
    # Define custom data lines for transaction data and features
    lines = (
        'transaction_price',      # Individual transaction price
        'sol_amount',            # SOL amount in transaction
        'is_buy',                # 1 if buy, 0 if sell
        'trader_id',             # Trader wallet address hash
        'transaction_size_category', # 0=small, 1=medium, 2=big, 3=whale
        'cumulative_volume',     # Running volume total
        'transaction_count',     # Running transaction count
        'buy_pressure',          # Recent buy ratio
        'volume_momentum',       # Volume trend indicator
        
        # ML Features from WindowFeatureCalculator (69 features total)
        # 30s window features (23 features)
        'total_volume_30s', 'buy_volume_30s', 'sell_volume_30s', 'buy_ratio_30s', 'volume_imbalance_30s', 'avg_txn_size_30s',
        'total_txns_30s', 'buy_txns_30s', 'sell_txns_30s', 'txn_buy_ratio_30s', 'txn_flow_imbalance_30s',
        'unique_traders_30s', 'unique_buyers_30s', 'unique_sellers_30s', 'trader_buy_ratio_30s',
        'small_txn_ratio_30s', 'medium_txn_ratio_30s', 'big_txn_ratio_30s', 'whale_txn_ratio_30s',
        'volume_per_trader_30s', 'volume_concentration_30s', 'volume_mean_30s', 'volume_std_30s',
        
        # 60s window features (23 features)
        'total_volume_60s', 'buy_volume_60s', 'sell_volume_60s', 'buy_ratio_60s', 'volume_imbalance_60s', 'avg_txn_size_60s',
        'total_txns_60s', 'buy_txns_60s', 'sell_txns_60s', 'txn_buy_ratio_60s', 'txn_flow_imbalance_60s',
        'unique_traders_60s', 'unique_buyers_60s', 'unique_sellers_60s', 'trader_buy_ratio_60s',
        'small_txn_ratio_60s', 'medium_txn_ratio_60s', 'big_txn_ratio_60s', 'whale_txn_ratio_60s',
        'volume_per_trader_60s', 'volume_concentration_60s', 'volume_mean_60s', 'volume_std_60s',
        
        # 120s window features (23 features)
        'total_volume_120s', 'buy_volume_120s', 'sell_volume_120s', 'buy_ratio_120s', 'volume_imbalance_120s', 'avg_txn_size_120s',
        'total_txns_120s', 'buy_txns_120s', 'sell_txns_120s', 'txn_buy_ratio_120s', 'txn_flow_imbalance_120s',
        'unique_traders_120s', 'unique_buyers_120s', 'unique_sellers_120s', 'trader_buy_ratio_120s',
        'small_txn_ratio_120s', 'medium_txn_ratio_120s', 'big_txn_ratio_120s', 'whale_txn_ratio_120s',
        'volume_per_trader_120s', 'volume_concentration_120s', 'volume_mean_120s', 'volume_std_120s',
    )
    
    params = (
        ('transaction_data', None),     # DataFrame with transaction data
        ('aggregation_window', 60),     # Seconds to aggregate features over
        ('min_transactions', 5),        # Minimum transactions to generate signal
        ('min_lookback_seconds', 120),  # Minimum lookback buffer (matches feature extraction)
        ('min_forward_seconds', 300),   # Minimum forward buffer for target calculation (5 minutes)
        ('feature_start_time', None),   # Explicit feature data start time (overrides calculated boundaries)
        ('feature_end_time', None),     # Explicit feature data end time (overrides calculated boundaries)
        ('datetime_col', 'block_timestamp'),
        ('price_col', 'price'),
        ('volume_col', 'sol_amount'), 
        ('direction_col', 'is_buy'),
        ('trader_col', 'swapper'),
        ('succeed_col', 'succeeded'),
    )
    
    def __init__(self):
        super().__init__()
        
        if self.params.transaction_data is None:
            raise ValueError("transaction_data parameter is required")
        
        # Initialize feature calculator if available
        self.feature_calculator = WindowFeatureCalculator() if WindowFeatureCalculator else None
        if not self.feature_calculator:
            print("Warning: Feature calculation disabled. WindowFeatureCalculator not available.")
        
        # Prepare transaction data
        self.df = self.params.transaction_data.copy()
        self._prepare_data()
        
        # Initialize state tracking
        self.current_idx = 0
        self.transaction_buffer = []
        self.volume_buffer = []
        self.last_timestamp = None
        self.skipped_early_windows = 0
        
    def _prepare_data(self):
        """Prepare and validate transaction data"""
        print(f"📊 Preparing {len(self.df)} transactions for backtesting...")
        
        # Ensure required columns exist
        required_cols = [
            self.params.datetime_col,
            self.params.volume_col,
            self.params.direction_col
        ]
        
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.params.datetime_col]):
            self.df[self.params.datetime_col] = pd.to_datetime(self.df[self.params.datetime_col])
        
        # Sort by timestamp
        self.df = self.df.sort_values(self.params.datetime_col).reset_index(drop=True)
        
        # Filter successful transactions only
        if self.params.succeed_col in self.df.columns:
            self.df = self.df[self.df[self.params.succeed_col] == True].reset_index(drop=True)
        
        # Calculate transaction price (if not provided)
        if self.params.price_col not in self.df.columns:
            # Estimate price from transaction ratio (placeholder)
            self.df['price'] = 1.0  # Would need actual price calculation
        
        # Categorize transaction sizes
        self.df['txn_size_category'] = self._categorize_transaction_sizes()
        
        # Add trader ID hash (for privacy)
        if self.params.trader_col in self.df.columns:
            self.df['trader_hash'] = self.df[self.params.trader_col].apply(lambda x: hash(str(x)) % 10000)
        else:
            self.df['trader_hash'] = 0
        
        # Generate sample timestamps using exact feature extraction logic
        self._generate_sample_timestamps()
        
        print(f"✅ Prepared data: {len(self.df)} valid transactions")
        print(f"   Date range: {self.df[self.params.datetime_col].min()} to {self.df[self.params.datetime_col].max()}")
        print(f"   Buy transactions: {self.df[self.params.direction_col].sum()}")
        print(f"   Sell transactions: {(~self.df[self.params.direction_col]).sum()}")
        print(f"   Generated {len(self.sample_timestamps)} sample timestamps")
        if len(self.sample_timestamps) > 0:
            print(f"   Sample range: {self.sample_timestamps[0]} to {self.sample_timestamps[-1]}")
    
    def _categorize_transaction_sizes(self) -> np.ndarray:
        """Categorize transactions by SOL amount"""
        amounts = self.df[self.params.volume_col]
        
        categories = np.zeros(len(amounts), dtype=int)
        categories[amounts <= 1] = 0      # Small: 0-1 SOL
        categories[(amounts > 1) & (amounts <= 10)] = 1    # Medium: 1-10 SOL
        categories[(amounts > 10) & (amounts <= 100)] = 2  # Big: 10-100 SOL
        categories[amounts > 100] = 3     # Whale: 100+ SOL
        
        return categories
    
    def _generate_sample_timestamps(self):
        """Generate sample timestamps using exact feature extraction logic"""
        # Use explicit boundaries if provided, otherwise calculate from data
        if self.params.feature_start_time is not None and self.params.feature_end_time is not None:
            sampling_start = self.params.feature_start_time
            sampling_end = self.params.feature_end_time
        else:
            # Calculate boundaries based on transaction data with buffers
            data_start = self.df[self.params.datetime_col].min()
            data_end = self.df[self.params.datetime_col].max()
            sampling_start = data_start + timedelta(seconds=self.params.min_lookback_seconds)
            sampling_end = data_end - timedelta(seconds=self.params.min_forward_seconds)
        
        # Generate timestamps every 60 seconds (SAMPLING_INTERVAL_SECONDS)
        self.sample_timestamps = []
        current_time = sampling_start
        
        while current_time <= sampling_end:
            self.sample_timestamps.append(current_time)
            current_time += timedelta(seconds=self.params.aggregation_window)
        
        # Reset current index to iterate through sample timestamps
        self.current_sample_idx = 0
    
    def _load(self):
        """
        Load next data point (called by Backtrader)
        Uses pre-generated sample timestamps to match feature extraction exactly
        """
        # Check if we have more sample timestamps
        if self.current_sample_idx >= len(self.sample_timestamps):
            return False  # No more sample timestamps
        
        # Get current sample timestamp (this is our reference point)
        current_time = self.sample_timestamps[self.current_sample_idx]
        
        # Define lookback window for feature calculation (120 seconds back from current_time)
        lookback_seconds = 120  # Match feature extraction lookback
        window_start = current_time - timedelta(seconds=lookback_seconds)
        window_end = current_time  # Features use data UP TO the sample timestamp
        
        # Collect all transactions in the lookback window using DataFrame filtering
        mask = (self.df[self.params.datetime_col] >= window_start) & (self.df[self.params.datetime_col] < window_end)
        window_transactions_df = self.df[mask].copy()
        
        # Calculate ML features using WindowFeatureCalculator
        if self.feature_calculator:
            ml_features = self.feature_calculator.calculate_features(window_transactions_df, current_time)
        else:
            ml_features = self._get_default_features()
        
        # Convert to legacy format for backward compatibility
        features = self._convert_ml_features_to_legacy(ml_features, window_transactions_df)
        
        # Set Backtrader data lines
        # Ensure current_time maintains its timezone info for bt.date2num
        if hasattr(current_time, 'tz_localize') and current_time.tz is None:
            # If timezone-naive, assume it matches the input data timezone
            if len(self.df) > 0 and hasattr(self.df.iloc[0][self.params.datetime_col], 'tz'):
                source_tz = self.df.iloc[0][self.params.datetime_col].tz
                if source_tz is not None:
                    current_time = current_time.tz_localize(source_tz)
        
        self.lines.datetime[0] = bt.date2num(current_time)
        
        # Set basic OHLCV data for onchain broker
        price = features['avg_price']
        volume = features['total_volume']
        
        # For onchain data, we use actual transaction price as all OHLC values
        # since transactions execute at specific prices, not ranges
        self.lines.open[0] = price
        self.lines.high[0] = price
        self.lines.low[0] = price  
        self.lines.close[0] = price
        self.lines.volume[0] = volume
        
        # Transaction-specific lines (legacy)
        self.lines.transaction_price[0] = features['avg_price']
        self.lines.sol_amount[0] = features['total_volume']
        self.lines.is_buy[0] = features['buy_ratio']
        self.lines.trader_id[0] = features['unique_traders']
        self.lines.transaction_size_category[0] = features['whale_ratio']
        self.lines.cumulative_volume[0] = features['cumulative_volume']
        self.lines.transaction_count[0] = len(window_transactions_df)
        self.lines.buy_pressure[0] = features['buy_pressure']
        self.lines.volume_momentum[0] = features['volume_momentum']
        
        # Set all ML feature lines
        self._set_ml_feature_lines(ml_features)
        
        # Update tracking
        self.current_sample_idx += 1
        self.last_timestamp = current_time
        
        return True
    
    
    def _calculate_buy_pressure(self, df_window: pd.DataFrame) -> float:
        """Calculate buy pressure indicator"""
        if len(df_window) == 0:
            return 0.0  # Return 0.0 when no data (actual calculated value)
        
        # Weight recent transactions more heavily
        weights = np.linspace(0.5, 1.0, len(df_window))
        buy_values = df_window[self.params.direction_col].values.astype(float)
        
        weighted_buy_pressure = np.average(buy_values, weights=weights)
        return weighted_buy_pressure
    
    def _calculate_volume_momentum(self) -> float:
        """Calculate volume momentum over recent windows"""
        if len(self.volume_buffer) < 2:
            return 0.0
        
        # Keep only recent windows
        if len(self.volume_buffer) > 10:
            self.volume_buffer = self.volume_buffer[-10:]
        
        # Calculate momentum as recent average vs earlier average
        if len(self.volume_buffer) >= 4:
            recent_avg = np.mean(self.volume_buffer[-2:])
            earlier_avg = np.mean(self.volume_buffer[-4:-2])
            
            if earlier_avg > 0:
                momentum = (recent_avg - earlier_avg) / earlier_avg
                return np.clip(momentum, -1.0, 1.0)
        
        return 0.0
    
    def _convert_ml_features_to_legacy(self, ml_features: Dict, window_df: pd.DataFrame) -> Dict[str, float]:
        """Convert ML features to legacy format for backward compatibility."""
        if window_df.empty:
            return {
                'total_volume': 0.0,
                'avg_price': 1.0,
                'buy_ratio': 0.0,  # 0.0 when no data (actual calculated value)
                'unique_traders': 0,
                'whale_ratio': 0.0,
                'cumulative_volume': getattr(self, '_cumulative_volume', 0),
                'buy_pressure': 0.0,  # 0.0 when no data (actual calculated value)
                'volume_momentum': 0.0
            }
        
        # Extract key metrics from 60s window features
        total_volume = ml_features.get('total_volume_60s', 0.0)
        buy_ratio = ml_features.get('buy_ratio_60s', 0.0)  # Use 0.0 when no data
        unique_traders = ml_features.get('unique_traders_60s', 0)
        whale_ratio = ml_features.get('whale_txn_ratio_60s', 0.0)
        
        # Calculate cumulative volume
        cumulative_volume = getattr(self, '_cumulative_volume', 0) + total_volume
        self._cumulative_volume = cumulative_volume
        
        # Calculate legacy metrics
        avg_price = window_df['price'].mean() if 'price' in window_df and len(window_df) > 0 else 1.0
        
        # Store volume for momentum calculation
        self.volume_buffer.append(total_volume)
        buy_pressure = self._calculate_buy_pressure(window_df)
        volume_momentum = self._calculate_volume_momentum()
        
        return {
            'total_volume': total_volume,
            'avg_price': avg_price,
            'buy_ratio': buy_ratio,
            'unique_traders': unique_traders,
            'whale_ratio': whale_ratio,
            'cumulative_volume': cumulative_volume,
            'buy_pressure': buy_pressure,
            'volume_momentum': volume_momentum
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default ML features when feature calculator is not available."""
        default_features = {'sample_timestamp': None, 'total_transactions': 0}
        
        # Generate default values for all 69 features
        for window in [30, 60, 120]:
            suffix = f"_{window}s"
            default_features.update({
                f'total_volume{suffix}': 0.0,
                f'buy_volume{suffix}': 0.0,
                f'sell_volume{suffix}': 0.0,
                f'buy_ratio{suffix}': 0.0,  # 0.0 when no data (actual calculated value)
                f'volume_imbalance{suffix}': 0.0,
                f'avg_txn_size{suffix}': 0.0,
                f'total_txns{suffix}': 0.0,
                f'buy_txns{suffix}': 0.0,
                f'sell_txns{suffix}': 0.0,
                f'txn_buy_ratio{suffix}': 0.0,  # 0.0 when no data (actual calculated value)
                f'txn_flow_imbalance{suffix}': 0.0,
                f'unique_traders{suffix}': 0.0,
                f'unique_buyers{suffix}': 0.0,
                f'unique_sellers{suffix}': 0.0,
                f'trader_buy_ratio{suffix}': 0.0,  # 0.0 when no data (actual calculated value)
                f'small_txn_ratio{suffix}': 0.0,
                f'medium_txn_ratio{suffix}': 0.0,
                f'big_txn_ratio{suffix}': 0.0,
                f'whale_txn_ratio{suffix}': 0.0,
                f'volume_per_trader{suffix}': 0.0,
                f'volume_concentration{suffix}': 0.0,
                f'volume_mean{suffix}': 0.0,
                f'volume_std{suffix}': 0.0
            })
        
        return default_features
    
    def _set_ml_feature_lines(self, ml_features: Dict[str, float]):
        """Set all ML feature data lines."""
        # Get all ML feature line names (exclude legacy transaction lines)
        ml_line_names = [
            'total_volume_30s', 'buy_volume_30s', 'sell_volume_30s', 'buy_ratio_30s', 'volume_imbalance_30s', 'avg_txn_size_30s',
            'total_txns_30s', 'buy_txns_30s', 'sell_txns_30s', 'txn_buy_ratio_30s', 'txn_flow_imbalance_30s',
            'unique_traders_30s', 'unique_buyers_30s', 'unique_sellers_30s', 'trader_buy_ratio_30s',
            'small_txn_ratio_30s', 'medium_txn_ratio_30s', 'big_txn_ratio_30s', 'whale_txn_ratio_30s',
            'volume_per_trader_30s', 'volume_concentration_30s', 'volume_mean_30s', 'volume_std_30s',
            
            'total_volume_60s', 'buy_volume_60s', 'sell_volume_60s', 'buy_ratio_60s', 'volume_imbalance_60s', 'avg_txn_size_60s',
            'total_txns_60s', 'buy_txns_60s', 'sell_txns_60s', 'txn_buy_ratio_60s', 'txn_flow_imbalance_60s',
            'unique_traders_60s', 'unique_buyers_60s', 'unique_sellers_60s', 'trader_buy_ratio_60s',
            'small_txn_ratio_60s', 'medium_txn_ratio_60s', 'big_txn_ratio_60s', 'whale_txn_ratio_60s',
            'volume_per_trader_60s', 'volume_concentration_60s', 'volume_mean_60s', 'volume_std_60s',
            
            'total_volume_120s', 'buy_volume_120s', 'sell_volume_120s', 'buy_ratio_120s', 'volume_imbalance_120s', 'avg_txn_size_120s',
            'total_txns_120s', 'buy_txns_120s', 'sell_txns_120s', 'txn_buy_ratio_120s', 'txn_flow_imbalance_120s',
            'unique_traders_120s', 'unique_buyers_120s', 'unique_sellers_120s', 'trader_buy_ratio_120s',
            'small_txn_ratio_120s', 'medium_txn_ratio_120s', 'big_txn_ratio_120s', 'whale_txn_ratio_120s',
            'volume_per_trader_120s', 'volume_concentration_120s', 'volume_mean_120s', 'volume_std_120s'
        ]
        
        # Set each ML feature line
        for line_name in ml_line_names:
            if hasattr(self.lines, line_name):
                feature_value = ml_features.get(line_name, 0.0)
                line_obj = getattr(self.lines, line_name)
                line_obj[0] = feature_value

