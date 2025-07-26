#!/usr/bin/env python3
"""
Window Feature Calculator for Classification Forward Model
Calculates all 69 features from a window of transaction data
Based on solana/docs/classification_forward_feature_engineering.md
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class WindowFeatureCalculator:
    """
    Calculates classification features for transaction windows.
    
    Implements the exact feature engineering logic from the SQL-based
    batch_feature_extraction.sql for real-time backtesting.
    """
    
    # Transaction size categories (SOL amounts)
    SMALL_THRESHOLD = 1.0      # 0-1 SOL (retail trades)
    MEDIUM_THRESHOLD = 10.0    # 1-10 SOL (small investors)
    BIG_THRESHOLD = 100.0      # 10-100 SOL (serious traders)
    # WHALE: 100+ SOL (institutional/whale activity)
    
    def __init__(self):
        """Initialize the feature calculator."""
        pass
    
    def calculate_features(self, 
                         transactions: pd.DataFrame, 
                         sample_timestamp: datetime,
                         windows: List[int] = [30, 60, 120]) -> Dict[str, Any]:
        """
        Calculate all features for given transaction data and sample timestamp.
        
        Args:
            transactions: DataFrame with transaction data
            sample_timestamp: Reference timestamp for feature calculation
            windows: List of lookback windows in seconds [30, 60, 120]
            
        Returns:
            Dictionary with all 69 features plus metadata
        """
        if transactions.empty:
            return self._get_zero_features(sample_timestamp, windows)
        
        # Prepare transaction data
        transactions = self._prepare_transactions(transactions)
        
        # Initialize feature dictionary
        features = {
            'sample_timestamp': sample_timestamp,
            'total_transactions': len(transactions)
        }
        
        # Calculate features for each window
        for window_seconds in windows:
            window_features = self._calculate_window_features(
                transactions, sample_timestamp, window_seconds
            )
            features.update(window_features)
        
        return features
    
    def _prepare_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Prepare transaction data with derived fields."""
        df = transactions.copy()
        
        # Ensure required columns exist
        required_columns = ['block_timestamp', 'is_buy', 'sol_amount', 'swapper']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from transaction data")
        
        # Add transaction size categories
        df['size_category'] = self._categorize_transaction_sizes(df['sol_amount'])
        
        # Ensure is_buy is boolean
        df['is_buy'] = df['is_buy'].astype(bool)
        
        # Add is_sell flag
        df['is_sell'] = ~df['is_buy']
        
        return df
    
    def _categorize_transaction_sizes(self, amounts: pd.Series) -> pd.Series:
        """Categorize transactions by SOL amount."""
        categories = pd.Series('whale', index=amounts.index)
        categories[amounts < self.SMALL_THRESHOLD] = 'small'  # Changed from <= to < for 1.0 SOL boundary
        categories[(amounts >= self.SMALL_THRESHOLD) & (amounts < self.MEDIUM_THRESHOLD)] = 'medium'
        categories[(amounts >= self.MEDIUM_THRESHOLD) & (amounts < self.BIG_THRESHOLD)] = 'big'
        return categories
    
    def _calculate_window_features(self, 
                                 transactions: pd.DataFrame, 
                                 sample_timestamp: datetime, 
                                 window_seconds: int) -> Dict[str, float]:
        """Calculate features for a specific window."""
        # Filter transactions to window
        window_start = sample_timestamp - timedelta(seconds=window_seconds)
        window_end = sample_timestamp
        
        window_data = transactions[
            (transactions['block_timestamp'] >= window_start) & 
            (transactions['block_timestamp'] < window_end)
        ].copy()
        
        if window_data.empty:
            return self._get_zero_window_features(window_seconds)
        
        window_suffix = f"_{window_seconds}s"
        features = {}
        
        # 1. Volume Flow Features (6 features)
        features.update(self._calculate_volume_features(window_data, window_suffix))
        
        # 2. Transaction Flow Features (5 features)
        features.update(self._calculate_transaction_features(window_data, window_suffix))
        
        # 3. Trader Behavior Features (4 features)
        features.update(self._calculate_trader_features(window_data, window_suffix))
        
        # 4. Position Size Distribution Features (4 features)
        features.update(self._calculate_size_features(window_data, window_suffix))
        
        # 5. Market Microstructure Features (4 features)
        features.update(self._calculate_microstructure_features(window_data, window_suffix))
        
        return features
    
    def _calculate_volume_features(self, window_data: pd.DataFrame, suffix: str) -> Dict[str, float]:
        """Calculate volume flow features."""
        total_volume = window_data['sol_amount'].sum()
        buy_volume = window_data[window_data['is_buy']]['sol_amount'].sum()
        sell_volume = window_data[window_data['is_sell']]['sol_amount'].sum()
        
        # Handle division by zero - use actual calculated values, not defaults
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.0  # 0.0 when no volume, not 0.5
        volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
        avg_txn_size = total_volume / len(window_data) if len(window_data) > 0 else 0.0
        
        return {
            f'total_volume{suffix}': total_volume,
            f'buy_volume{suffix}': buy_volume,
            f'sell_volume{suffix}': sell_volume,
            f'buy_ratio{suffix}': buy_ratio,
            f'volume_imbalance{suffix}': volume_imbalance,
            f'avg_txn_size{suffix}': avg_txn_size
        }
    
    def _calculate_transaction_features(self, window_data: pd.DataFrame, suffix: str) -> Dict[str, float]:
        """Calculate transaction flow features."""
        total_txns = len(window_data)
        buy_txns = window_data['is_buy'].sum()
        sell_txns = window_data['is_sell'].sum()
        
        # Handle division by zero - use actual calculated values, not defaults
        txn_buy_ratio = buy_txns / total_txns if total_txns > 0 else 0.0  # 0.0 when no transactions, not 0.5
        txn_flow_imbalance = (buy_txns - sell_txns) / total_txns if total_txns > 0 else 0.0
        
        return {
            f'total_txns{suffix}': float(total_txns),
            f'buy_txns{suffix}': float(buy_txns),
            f'sell_txns{suffix}': float(sell_txns),
            f'txn_buy_ratio{suffix}': txn_buy_ratio,
            f'txn_flow_imbalance{suffix}': txn_flow_imbalance
        }
    
    def _calculate_trader_features(self, window_data: pd.DataFrame, suffix: str) -> Dict[str, float]:
        """Calculate trader behavior features."""
        unique_traders = window_data['swapper'].nunique()
        unique_buyers = window_data[window_data['is_buy']]['swapper'].nunique()
        unique_sellers = window_data[window_data['is_sell']]['swapper'].nunique()
        
        # Handle division by zero - use actual calculated values, not defaults
        trader_buy_ratio = unique_buyers / unique_traders if unique_traders > 0 else 0.0  # 0.0 when no traders, not 0.5
        
        return {
            f'unique_traders{suffix}': float(unique_traders),
            f'unique_buyers{suffix}': float(unique_buyers),
            f'unique_sellers{suffix}': float(unique_sellers),
            f'trader_buy_ratio{suffix}': trader_buy_ratio
        }
    
    def _calculate_size_features(self, window_data: pd.DataFrame, suffix: str) -> Dict[str, float]:
        """Calculate position size distribution features."""
        total_txns = len(window_data)
        
        if total_txns == 0:
            return {
                f'small_txn_ratio{suffix}': 0.0,
                f'medium_txn_ratio{suffix}': 0.0,
                f'big_txn_ratio{suffix}': 0.0,
                f'whale_txn_ratio{suffix}': 0.0
            }
        
        size_counts = window_data['size_category'].value_counts()
        
        return {
            f'small_txn_ratio{suffix}': size_counts.get('small', 0) / total_txns,
            f'medium_txn_ratio{suffix}': size_counts.get('medium', 0) / total_txns,
            f'big_txn_ratio{suffix}': size_counts.get('big', 0) / total_txns,
            f'whale_txn_ratio{suffix}': size_counts.get('whale', 0) / total_txns
        }
    
    def _calculate_microstructure_features(self, window_data: pd.DataFrame, suffix: str) -> Dict[str, float]:
        """Calculate market microstructure features."""
        unique_traders = window_data['swapper'].nunique()
        total_volume = window_data['sol_amount'].sum()
        
        # Volume per trader
        volume_per_trader = total_volume / unique_traders if unique_traders > 0 else 0.0
        
        # Volume statistics
        amounts = window_data['sol_amount']
        volume_mean = amounts.mean() if len(amounts) > 0 else 0.0
        volume_std = amounts.std(ddof=0) if len(amounts) > 1 else 0.0  # Use population std (ddof=0) to match SQL STDDEV_POP
        
        # Volume concentration (coefficient of variation) - matches SQL formula
        volume_concentration = volume_std / (volume_mean + 1e-10) if volume_mean > 0 else 0.0
        
        return {
            f'volume_per_trader{suffix}': volume_per_trader,
            f'volume_concentration{suffix}': volume_concentration,
            f'volume_mean{suffix}': volume_mean,
            f'volume_std{suffix}': volume_std
        }
    
    def _get_zero_window_features(self, window_seconds: int) -> Dict[str, float]:
        """Return zero features for empty windows."""
        suffix = f"_{window_seconds}s"
        
        return {
            # Volume features
            f'total_volume{suffix}': 0.0,
            f'buy_volume{suffix}': 0.0,
            f'sell_volume{suffix}': 0.0,
            f'buy_ratio{suffix}': 0.0,  # 0.0 when no data (no buy volume / no total volume)
            f'volume_imbalance{suffix}': 0.0,
            f'avg_txn_size{suffix}': 0.0,
            
            # Transaction features
            f'total_txns{suffix}': 0.0,
            f'buy_txns{suffix}': 0.0,
            f'sell_txns{suffix}': 0.0,
            f'txn_buy_ratio{suffix}': 0.0,  # 0.0 when no data (no buy transactions / no total transactions)
            f'txn_flow_imbalance{suffix}': 0.0,
            
            # Trader features
            f'unique_traders{suffix}': 0.0,
            f'unique_buyers{suffix}': 0.0,
            f'unique_sellers{suffix}': 0.0,
            f'trader_buy_ratio{suffix}': 0.0,  # 0.0 when no data (no unique buyers / no unique traders)
            
            # Size features
            f'small_txn_ratio{suffix}': 0.0,
            f'medium_txn_ratio{suffix}': 0.0,
            f'big_txn_ratio{suffix}': 0.0,
            f'whale_txn_ratio{suffix}': 0.0,
            
            # Microstructure features
            f'volume_per_trader{suffix}': 0.0,
            f'volume_concentration{suffix}': 0.0,
            f'volume_mean{suffix}': 0.0,
            f'volume_std{suffix}': 0.0
        }
    
    def _get_zero_features(self, sample_timestamp: datetime, windows: List[int]) -> Dict[str, Any]:
        """Return zero features for completely empty transaction data."""
        features = {
            'sample_timestamp': sample_timestamp,
            'total_transactions': 0
        }
        
        for window_seconds in windows:
            features.update(self._get_zero_window_features(window_seconds))
        
        return features
    
    def get_feature_names(self, windows: List[int] = [30, 60, 120]) -> List[str]:
        """Get list of all feature names (excluding metadata)."""
        feature_names = []
        
        base_features = [
            'total_volume', 'buy_volume', 'sell_volume', 'buy_ratio', 'volume_imbalance', 'avg_txn_size',
            'total_txns', 'buy_txns', 'sell_txns', 'txn_buy_ratio', 'txn_flow_imbalance',
            'unique_traders', 'unique_buyers', 'unique_sellers', 'trader_buy_ratio',
            'small_txn_ratio', 'medium_txn_ratio', 'big_txn_ratio', 'whale_txn_ratio',
            'volume_per_trader', 'volume_concentration', 'volume_mean', 'volume_std'
        ]
        
        for window_seconds in windows:
            suffix = f"_{window_seconds}s"
            for base_feature in base_features:
                feature_names.append(f"{base_feature}{suffix}")
        
        return feature_names


def test_feature_calculator():
    """Test the feature calculator against real DuckDB data."""
    import duckdb
    
    print("🧪 Testing Window Feature Calculator with Real Data")
    print("=" * 60)
    
    # Connect to DuckDB
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Pick a test coin and get one sample
    coin_id = "8xPe3DMr52oYAkCBk57ZeQE1h5zBkWNk37eE4J8spump"
    print(f"Testing coin: {coin_id}")
    
    # Get one feature sample as ground truth
    feature_query = f"""
    SELECT *
    FROM classification_forward_features
    WHERE coin_id = '{coin_id}'
    ORDER BY sample_timestamp
    LIMIT 1
    """
    
    feature_sample = conn.execute(feature_query).fetchdf()
    if len(feature_sample) == 0:
        print("❌ No feature samples found")
        return
    
    test_sample = feature_sample.iloc[0]
    sample_timestamp = test_sample['sample_timestamp']
    print(f"Testing timestamp: {sample_timestamp}")
    
    # Get transaction data for this coin - ONLY SOL-related trades
    transaction_query = f"""
    SELECT 
        block_timestamp,
        swapper,
        CASE WHEN mint = swap_to_mint THEN 1 ELSE 0 END as is_buy,
        CASE 
            WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_from_amount
            WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_to_amount
            ELSE 0.0
        END as sol_amount
    FROM first_day_trades
    WHERE mint = '{coin_id}'
    AND succeeded = TRUE
    AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
         OR swap_to_mint = 'So11111111111111111111111111111111111111112')
    AND mint != 'So11111111111111111111111111111111111111112'
    ORDER BY block_timestamp
    """
    
    transactions_df = conn.execute(transaction_query).fetchdf()
    print(f"Loaded {len(transactions_df)} transactions")
    
    # Calculate features
    calculator = WindowFeatureCalculator()
    calculated_features = calculator.calculate_features(transactions_df, sample_timestamp)
    
    print(f"\n📊 FEATURE COMPARISON:")
    print("=" * 60)
    
    # Compare all features across all windows
    test_features = []
    for window in [30, 60, 120]:
        window_features = [
            f'total_volume_{window}s', f'buy_volume_{window}s', f'sell_volume_{window}s', 
            f'buy_ratio_{window}s', f'volume_imbalance_{window}s', f'total_txns_{window}s',
            f'buy_txns_{window}s', f'sell_txns_{window}s', f'txn_buy_ratio_{window}s',
            f'txn_flow_imbalance_{window}s', f'unique_traders_{window}s', f'unique_buyers_{window}s',
            f'unique_sellers_{window}s', f'trader_buy_ratio_{window}s', f'avg_txn_size_{window}s',
            f'volume_per_trader_{window}s', f'volume_concentration_{window}s', f'volume_mean_{window}s',
            f'volume_std_{window}s', f'small_txn_ratio_{window}s', f'medium_txn_ratio_{window}s',
            f'big_txn_ratio_{window}s', f'whale_txn_ratio_{window}s'
        ]
        test_features.extend(window_features)
    
    matches = 0
    for feature in test_features:
        if feature in test_sample.index and feature in calculated_features:
            ground_truth = test_sample[feature]
            calculated = calculated_features[feature]
            
            # Compare with tolerance
            if pd.isna(ground_truth) and pd.isna(calculated):
                match = True
                diff = 0.0
            elif pd.isna(ground_truth) or pd.isna(calculated):
                match = False
                diff = float('inf')
            else:
                diff = abs(float(ground_truth) - float(calculated))
                match = diff < 1e-6 or (abs(diff / max(abs(float(ground_truth)), 1e-6)) < 1e-4)
            
            status = "✅" if match else "❌"
            print(f"{status} {feature}")
            print(f"    Ground truth: {ground_truth}")
            print(f"    Calculated:   {calculated}")
            if not match and diff != float('inf'):
                print(f"    Difference:   {diff:.8f}")
            
            if match:
                matches += 1
    
    accuracy = matches / len(test_features) * 100
    print(f"\n📈 Accuracy: {matches}/{len(test_features)} ({accuracy:.1f}%)")
    
    # Debug transaction size categorization for 120s window
    print(f"\n🔍 DEBUGGING 120s TRANSACTION SIZE CATEGORIZATION:")
    print("=" * 60)
    
    window_start_120s = sample_timestamp - pd.Timedelta(seconds=120)
    window_end_120s = sample_timestamp
    
    window_120s_data = transactions_df[
        (transactions_df['block_timestamp'] >= window_start_120s) & 
        (transactions_df['block_timestamp'] < window_end_120s)
    ].copy()
    
    if len(window_120s_data) > 0:
        window_120s_data['size_category'] = calculator._categorize_transaction_sizes(window_120s_data['sol_amount'])
        
        print(f"Total transactions in 120s window: {len(window_120s_data)}")
        print(f"Ground truth total_txns_120s: {test_sample['total_txns_120s']}")
        
        # Show each transaction with its categorization
        print(f"\nTransaction details:")
        small_count = 0
        medium_count = 0
        for i, (_, txn) in enumerate(window_120s_data.iterrows()):
            sol_amt = txn['sol_amount']
            category = txn['size_category']
            if category == 'small':
                small_count += 1
            elif category == 'medium':
                medium_count += 1
            print(f"  {i+1:2d}. {txn['block_timestamp']} | {sol_amt:10.8f} SOL | {category}")
        
        print(f"\nMy categorization:")
        print(f"  Small (<=1.0): {small_count} transactions")
        print(f"  Medium (1.0-10.0): {medium_count} transactions")
        
        print(f"\nGround truth from SQL:")
        expected_small = int(test_sample['small_txn_ratio_120s'] * test_sample['total_txns_120s'])
        expected_medium = int(test_sample['medium_txn_ratio_120s'] * test_sample['total_txns_120s'])
        print(f"  Small: {expected_small} transactions ({test_sample['small_txn_ratio_120s']:.6f} ratio)")
        print(f"  Medium: {expected_medium} transactions ({test_sample['medium_txn_ratio_120s']:.6f} ratio)")
        
        print(f"\nDifference:")
        print(f"  Small: {small_count - expected_small} extra in my calculation")
        print(f"  Medium: {medium_count - expected_medium} difference in my calculation")
    
    if accuracy > 95:
        print("\n🎉 Feature calculator is highly accurate!")
    elif accuracy > 80:
        print("\n✅ Feature calculator is mostly accurate")
    else:
        print("\n⚠️ Feature calculator needs adjustments")
    
    conn.close()


if __name__ == "__main__":
    test_feature_calculator()