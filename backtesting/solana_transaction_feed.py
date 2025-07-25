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


class SolanaTransactionFeed(bt.feeds.DataBase):
    """
    Custom data feed for processing individual Solana transactions
    Bypasses OHLCV aggregation to work with raw transaction data
    """
    
    # Define custom data lines for transaction data
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
    )
    
    params = (
        ('transaction_data', None),     # DataFrame with transaction data
        ('aggregation_window', 60),     # Seconds to aggregate features over
        ('min_transactions', 5),        # Minimum transactions to generate signal
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
        
        # Initialize state tracking
        self.current_idx = 0
        self.transaction_buffer = []
        self.volume_buffer = []
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
        
        print(f"✅ Prepared data: {len(self.df)} valid transactions")
        print(f"   Date range: {self.df[self.params.datetime_col].min()} to {self.df[self.params.datetime_col].max()}")
        print(f"   Buy transactions: {self.df[self.params.direction_col].sum()}")
        print(f"   Sell transactions: {(~self.df[self.params.direction_col]).sum()}")
    
    def _categorize_transaction_sizes(self) -> np.ndarray:
        """Categorize transactions by SOL amount"""
        amounts = self.df[self.params.volume_col]
        
        categories = np.zeros(len(amounts), dtype=int)
        categories[amounts <= 1] = 0      # Small: 0-1 SOL
        categories[(amounts > 1) & (amounts <= 10)] = 1    # Medium: 1-10 SOL
        categories[(amounts > 10) & (amounts <= 100)] = 2  # Big: 10-100 SOL
        categories[amounts > 100] = 3     # Whale: 100+ SOL
        
        return categories
    
    def _load(self):
        """
        Load next data point (called by Backtrader)
        This processes transactions in time windows rather than individual ticks
        """
        if self.current_idx >= len(self.df):
            return False  # No more data
        
        # Get current timestamp window
        current_time = self.df.iloc[self.current_idx][self.params.datetime_col]
        window_end = current_time + timedelta(seconds=self.params.aggregation_window)
        
        # Collect all transactions in this time window
        window_transactions = []
        temp_idx = self.current_idx
        
        while (temp_idx < len(self.df) and 
               self.df.iloc[temp_idx][self.params.datetime_col] < window_end):
            window_transactions.append(self.df.iloc[temp_idx])
            temp_idx += 1
        
        # Skip if not enough transactions
        if len(window_transactions) < self.params.min_transactions:
            self.current_idx = temp_idx
            if self.current_idx >= len(self.df):
                return False
            return self._load()  # Try next window
        
        # Calculate aggregated features for this window
        features = self._calculate_window_features(window_transactions, current_time)
        
        # Set Backtrader data lines
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
        
        # Transaction-specific lines
        self.lines.transaction_price[0] = features['avg_price']
        self.lines.sol_amount[0] = features['total_volume']
        self.lines.is_buy[0] = features['buy_ratio']
        self.lines.trader_id[0] = features['unique_traders']
        self.lines.transaction_size_category[0] = features['whale_ratio']
        self.lines.cumulative_volume[0] = features['cumulative_volume']
        self.lines.transaction_count[0] = len(window_transactions)
        self.lines.buy_pressure[0] = features['buy_pressure']
        self.lines.volume_momentum[0] = features['volume_momentum']
        
        # Update tracking
        self.current_idx = temp_idx
        self.last_timestamp = current_time
        
        return True
    
    def _calculate_window_features(self, transactions: List, timestamp: datetime) -> Dict[str, float]:
        """Calculate features for a transaction window"""
        df_window = pd.DataFrame(transactions)
        
        # Basic volume and price features
        total_volume = df_window[self.params.volume_col].sum()
        avg_price = df_window['price'].mean() if 'price' in df_window else 1.0
        
        # Buy/sell analysis
        buy_mask = df_window[self.params.direction_col] == 1
        buy_volume = df_window[buy_mask][self.params.volume_col].sum()
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        
        # Trader analysis
        unique_traders = df_window['trader_hash'].nunique() if 'trader_hash' in df_window else len(transactions)
        
        # Transaction size analysis
        whale_txns = (df_window['txn_size_category'] == 3).sum()
        whale_ratio = whale_txns / len(transactions) if len(transactions) > 0 else 0
        
        # Calculate cumulative volume (running total)
        cumulative_volume = getattr(self, '_cumulative_volume', 0) + total_volume
        self._cumulative_volume = cumulative_volume
        
        # Buy pressure (recent trend)
        self.volume_buffer.append(total_volume)
        buy_pressure = self._calculate_buy_pressure(df_window)
        
        # Volume momentum
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
    
    def _calculate_buy_pressure(self, df_window: pd.DataFrame) -> float:
        """Calculate buy pressure indicator"""
        if len(df_window) == 0:
            return 0.5
        
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

