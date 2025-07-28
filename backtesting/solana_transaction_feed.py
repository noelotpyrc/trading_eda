#!/usr/bin/env python3
"""
Solana Transaction Data Feed
Clean data feed providing raw transaction data and OHLCV for Backtrader strategies
Focuses on data provision only - feature engineering handled by individual strategies
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os



class SolanaTransactionFeed(bt.feeds.DataBase):
    """
    Clean data feed for Solana transaction data
    
    Provides:
    - Standard OHLCV data calculated from transaction prices
    - Raw transaction data (price, volume, direction, trader, size category)
    - No feature engineering - strategies handle their own intelligence
    
    Price Scaling: Uses 1e9 scale factor (like Ethereum's Gwei) for numerical precision
    - Raw price: 0.0000000003 SOL/token → Scaled: 0.3 GSol/token
    - Prevents floating-point precision issues in P&L calculations
    """
    
    # Price scaling factor (like Ethereum's Gwei: 1e9)
    PRICE_SCALE_FACTOR = 1e9
    
    # Define custom data lines for raw transaction data only
    lines = (
        'transaction_price',      # Individual transaction price (scaled by 1e9)
        'sol_amount',            # SOL amount in transaction
        'is_buy',                # 1 if buy, 0 if sell
        'trader_id',             # Trader wallet address hash
        'transaction_size_category', # 0=small, 1=medium, 2=big, 3=whale
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
        
        
        # Prepare transaction data
        self.df = self.params.transaction_data.copy()
        self._prepare_data()
        
        # Add transaction prices based on SOL trades
        self._calculate_transaction_prices()
        
        # Initialize state tracking
        self.current_idx = 0
        self.last_timestamp = None
        
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
    
    def _calculate_transaction_prices(self):
        """Calculate price for each transaction based on SOL trades."""
        print(f"📊 Calculating transaction prices for {len(self.df)} transactions...")
        
        # SOL mint address
        SOL_MINT = 'So11111111111111111111111111111111111111112'
        
        def calculate_price_for_transaction(row):
            """Calculate price as SOL amount / token amount for SOL trades, scaled by 1e9."""
            # Check if this is a SOL trade
            if row['swap_from_mint'] == SOL_MINT:
                # SOL to token: price = SOL amount / token amount
                if row['swap_to_amount'] > 0:
                    raw_price = row['swap_from_amount'] / row['swap_to_amount']
                    return raw_price * self.PRICE_SCALE_FACTOR  # Scale by 1e9
                else:
                    return 0.0
            elif row['swap_to_mint'] == SOL_MINT:
                # Token to SOL: price = SOL amount / token amount  
                if row['swap_from_amount'] > 0:
                    raw_price = row['swap_to_amount'] / row['swap_from_amount']
                    return raw_price * self.PRICE_SCALE_FACTOR  # Scale by 1e9
                else:
                    return 0.0
            else:
                # Not a SOL trade
                return 0.0
        
        # Calculate price for each transaction
        self.df['transaction_price'] = self.df.apply(calculate_price_for_transaction, axis=1)
        
        # Filter out transactions with zero or invalid prices
        valid_price_mask = self.df['transaction_price'] > 0
        valid_count = valid_price_mask.sum()
        total_count = len(self.df)
        
        if total_count > 0:
            percentage = valid_count/total_count*100
            print(f"✅ Calculated prices for {valid_count}/{total_count} transactions ({percentage:.1f}%)")
        else:
            print("✅ Calculated prices for 0/0 transactions")
        
        # Don't override individual transaction prices with fallback
        # Let invalid transactions keep their 0.0 prices
    
    def _load(self):
        """
        Load next data point (called by Backtrader)
        Provides raw transaction data and OHLCV only
        """
        # Check if we have more sample timestamps
        if self.current_sample_idx >= len(self.sample_timestamps):
            return False  # No more sample timestamps
        
        # Get current sample timestamp (this is our reference point)
        current_time = self.sample_timestamps[self.current_sample_idx]
        
        # Define lookback window for aggregation (60 seconds back from current_time)
        lookback_seconds = self.params.aggregation_window
        window_start = current_time - timedelta(seconds=lookback_seconds)
        window_end = current_time
        
        # Collect all transactions in the window
        mask = (self.df[self.params.datetime_col] >= window_start) & (self.df[self.params.datetime_col] < window_end)
        window_transactions_df = self.df[mask].copy()
        
        # Calculate OHLCV from transactions for this sample timestamp
        ohlcv = self._calculate_ohlcv_from_transactions(current_time)
        
        # Set Backtrader data lines
        # Ensure current_time maintains its timezone info for bt.date2num
        if hasattr(current_time, 'tz_localize') and current_time.tz is None:
            # If timezone-naive, assume it matches the input data timezone
            if len(self.df) > 0 and hasattr(self.df.iloc[0][self.params.datetime_col], 'tz'):
                source_tz = self.df.iloc[0][self.params.datetime_col].tz
                if source_tz is not None:
                    current_time = current_time.tz_localize(source_tz)
        
        self.lines.datetime[0] = bt.date2num(current_time)
        
        # Set OHLCV data from calculated transaction prices
        self.lines.open[0] = ohlcv['open']
        self.lines.high[0] = ohlcv['high']
        self.lines.low[0] = ohlcv['low']
        self.lines.close[0] = ohlcv['close']
        self.lines.volume[0] = ohlcv['volume']
        
        # Set raw transaction data lines
        if not window_transactions_df.empty:
            # Use last transaction in window as representative values
            last_txn = window_transactions_df.iloc[-1]
            total_volume = window_transactions_df[self.params.volume_col].sum()
            avg_price = window_transactions_df['transaction_price'].mean() if 'transaction_price' in window_transactions_df else (1.0 * self.PRICE_SCALE_FACTOR)
            buy_ratio = window_transactions_df[self.params.direction_col].mean()
            unique_traders = window_transactions_df[self.params.trader_col].nunique() if self.params.trader_col in window_transactions_df else 1
            avg_size_category = window_transactions_df['txn_size_category'].mean() if 'txn_size_category' in window_transactions_df else 0
            
            self.lines.transaction_price[0] = avg_price
            self.lines.sol_amount[0] = total_volume
            self.lines.is_buy[0] = buy_ratio
            self.lines.trader_id[0] = unique_traders
            self.lines.transaction_size_category[0] = avg_size_category
        else:
            # No transactions in window - use OHLCV close price instead of hardcoded fallback
            self.lines.transaction_price[0] = ohlcv['close']  # Use intelligent fallback price
            self.lines.sol_amount[0] = 0.0
            self.lines.is_buy[0] = 0.0
            self.lines.trader_id[0] = 0
            self.lines.transaction_size_category[0] = 0
        
        # Update tracking
        self.current_sample_idx += 1
        self.last_timestamp = current_time
        
        return True
    
    def _calculate_ohlcv_from_transactions(self, sample_timestamp):
        """
        Calculate OHLCV from transactions around sample timestamp.
        
        Strategy:
        1. Use transactions from last 60s (sample_timestamp-60 to sample_timestamp)
        2. If no transactions in last 60s, fallback to close price from -120s to -60s
        3. Continue fallback pattern (-180s to -120s, etc.) until price found
        """
        # Primary window: last 60 seconds before sample timestamp
        primary_start = sample_timestamp - timedelta(seconds=60)
        primary_end = sample_timestamp
        
        primary_mask = (
            (self.df[self.params.datetime_col] >= primary_start) & 
            (self.df[self.params.datetime_col] < primary_end) &
            (self.df['transaction_price'] > 0)  # Only valid prices
        )
        primary_transactions = self.df[primary_mask].copy()
        
        if not primary_transactions.empty:
            # Use primary 60s window for OHLCV calculation
            return self._calculate_ohlcv_from_price_data(primary_transactions)
        
        # Fallback: Look for transactions in earlier 60s windows
        fallback_close_price = self._get_fallback_close_price(sample_timestamp)
        
        return {
            'open': fallback_close_price,
            'high': fallback_close_price,
            'low': fallback_close_price,
            'close': fallback_close_price,
            'volume': 0.0  # No volume in current window
        }
    
    def _calculate_ohlcv_from_price_data(self, transactions_df):
        """Calculate OHLCV from transactions with valid prices (scaled prices)."""
        if transactions_df.empty:
            # Use scaled fallback price (1.0 SOL/token * 1e9 = 1e9)
            fallback_price = 1.0 * self.PRICE_SCALE_FACTOR
            return {'open': fallback_price, 'high': fallback_price, 'low': fallback_price, 'close': fallback_price, 'volume': 0.0}
        
        # Sort by timestamp to get proper open/close sequence
        sorted_txns = transactions_df
        prices = sorted_txns['transaction_price']  # Already scaled prices
        volumes = sorted_txns[self.params.volume_col]
        
        return {
            'open': prices.iloc[0],          # First transaction price (scaled)
            'high': prices.max(),            # Highest transaction price (scaled)
            'low': prices.min(),             # Lowest transaction price (scaled)
            'close': prices.iloc[-1],        # Last transaction price (scaled)
            'volume': volumes.sum()          # Total SOL volume
        }
    
    def _get_fallback_close_price(self, sample_timestamp):
        """
        Get fallback close price from earlier time periods.
        Try -120s to -60s, then -180s to -120s, etc.
        """
        # Try fallback windows in 60s increments
        for lookback_start in [120, 180, 240, 300, 360]:  # 2min, 3min, 4min, 5min, 6min back
            lookback_end = lookback_start - 60
            
            fallback_start = sample_timestamp - timedelta(seconds=lookback_start)  
            fallback_end = sample_timestamp - timedelta(seconds=lookback_end)
            
            fallback_mask = (
                (self.df[self.params.datetime_col] >= fallback_start) & 
                (self.df[self.params.datetime_col] < fallback_end) &
                (self.df['transaction_price'] > 0)
            )
            fallback_transactions = self.df[fallback_mask]
            
            if not fallback_transactions.empty:
                # Return the close price (last transaction) from this window
                sorted_fallback = fallback_transactions.sort_values(self.params.datetime_col)
                return sorted_fallback['transaction_price'].iloc[-1]
        
        # Ultimate fallback if no transactions found in any window (scaled)
        return 1.0 * self.PRICE_SCALE_FACTOR

