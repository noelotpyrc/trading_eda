#!/usr/bin/env python3
"""
Test actual SolanaTransactionFeed + FeatureEngineeringStrategy alignment with feature data
Tests the new architecture: cleaned feed + separate feature engineering strategy
"""

import sys
import os
sys.path.append('/Users/noel/projects/trading_eda')

import pandas as pd
import duckdb
import backtrader as bt
from datetime import datetime, timedelta
from backtesting.solana_transaction_feed import SolanaTransactionFeed
from backtesting.strategies.feature_engineering_strategy import FeatureEngineeringStrategy
from backtesting.onchain_broker import setup_onchain_broker

def main():
    print("🧪 Actual Feed Alignment Test")
    print("=" * 40)
    
    # Test coin
    coin_id = "8xPe3DMr52oYAkCBk57ZeQE1h5zBkWNk37eE4J8spump"
    print(f"Testing coin: {coin_id}")
    
    # Connect to DuckDB
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Get transaction data
    print("\n1. Getting transaction data...")
    transaction_query = f"""
    SELECT 
        block_timestamp,
        mint,
        swapper,
        succeeded,
        CASE WHEN mint = swap_to_mint THEN 1 ELSE 0 END as is_buy,
        CASE 
            WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_from_amount
            WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_to_amount
            ELSE 0.0
        END as sol_amount,
        swap_from_amount,
        swap_to_amount,
        swap_from_mint,
        swap_to_mint
    FROM first_day_trades
    WHERE mint = '{coin_id}'
    AND succeeded = TRUE
    AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
         OR swap_to_mint = 'So11111111111111111111111111111111111111112')
    AND mint != 'So11111111111111111111111111111111111111112'
    ORDER BY block_timestamp
    """
    
    transaction_df = conn.execute(transaction_query).fetchdf()
    print(f"Found {len(transaction_df)} transactions")
    print(f"Time range: {transaction_df['block_timestamp'].min()} to {transaction_df['block_timestamp'].max()}")
    
    # Get feature data with more columns for comparison
    print("\n2. Getting feature data...")
    feature_query = f"""
    SELECT coin_id, sample_timestamp, 
           total_volume_30s, buy_ratio_30s, total_txns_30s, volume_concentration_30s, small_txn_ratio_30s,
           total_volume_60s, buy_ratio_60s, total_txns_60s, volume_concentration_60s, small_txn_ratio_60s,
           total_volume_120s, buy_ratio_120s, total_txns_120s, volume_concentration_120s, 
           small_txn_ratio_120s, medium_txn_ratio_120s, whale_txn_ratio_120s
    FROM classification_forward_features
    WHERE coin_id = '{coin_id}'
    ORDER BY sample_timestamp
    """
    
    feature_df = conn.execute(feature_query).fetchdf()
    print(f"Found {len(feature_df)} feature samples")
    if len(feature_df) > 0:
        print(f"Time range: {feature_df['sample_timestamp'].min()} to {feature_df['sample_timestamp'].max()}")
    
    # Create actual SolanaTransactionFeed + FeatureEngineeringStrategy
    print("\n3. Running new architecture: cleaned feed + feature engineering strategy...")
    
    # Create Cerebro
    cerebro = bt.Cerebro()
    
    # Create the cleaned transaction feed (no more feature engineering in feed)
    data_feed = SolanaTransactionFeed(
        transaction_data=transaction_df,
        aggregation_window=60,  # 1-minute windows to match DuckDB sampling
        datetime_col='block_timestamp',
        volume_col='sol_amount',
        direction_col='is_buy',
        trader_col='swapper',
        succeed_col='succeeded'
    )
    
    # Add feed to cerebro
    cerebro.adddata(data_feed)
    
    # Setup onchain broker
    setup_onchain_broker(cerebro, initial_cash=50000)
    
    # Capture features calculated by FeatureEngineeringStrategy
    calculated_features = []
    
    # Custom FeatureEngineeringStrategy that captures features without trading
    class TestFeatureEngineeringStrategy(FeatureEngineeringStrategy):
        def __init__(self):
            super().__init__()
            
        def next(self):
            # Run the normal feature engineering logic
            super().next()
            
            # Capture the calculated features
            if self.last_features:
                feature_record = {
                    'timestamp': self.datas[0].datetime.datetime(0),
                    **self.last_features
                }
                calculated_features.append(feature_record)
                
                # Debug first few records
                if len(calculated_features) <= 3:
                    print(f"  Debug {len(calculated_features)}: {feature_record['timestamp']}")
                    print(f"    Features - vol_60s: {self.last_features.get('total_volume_60s', 0):.6f} | buy_ratio_60s: {self.last_features.get('buy_ratio_60s', 0):.3f} | txns_60s: {self.last_features.get('total_txns_60s', 0)}")
                
                # Print progress
                if len(calculated_features) % 100 == 0:
                    print(f"  Calculated {len(calculated_features)} feature records...")
    
    # Add the FeatureEngineeringStrategy with no trading (high thresholds)
    cerebro.addstrategy(
        TestFeatureEngineeringStrategy,
        lookback_windows=[30, 60, 120],
        buy_ratio_threshold=1.0,  # Never buy
        volume_threshold=float('inf'),  # Never buy 
        trader_threshold=float('inf'),  # Never buy
        verbose=False,
        log_features=False
    )
    
    # Run the feed
    print("  Running feed to capture actual timestamps...")
    results = cerebro.run()
    
    # Extract timestamps from calculated features
    feature_timestamps = [record['timestamp'] for record in calculated_features]
    
    print(f"✅ Strategy calculated {len(feature_timestamps)} feature records")
    if len(feature_timestamps) > 0:
        print(f"Strategy time range: {min(feature_timestamps)} to {max(feature_timestamps)}")
    
    # Compare with DuckDB features
    print("\n4. Comparing calculated features with DuckDB ground truth...")
    
    if len(feature_df) == 0 or len(feature_timestamps) == 0:
        print("❌ No data to compare")
        return
    
    # Convert feature timestamps to UTC to match calculated timestamps
    duckdb_timestamps_utc = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    duckdb_timestamps_set = set(duckdb_timestamps_utc)
    calculated_timestamps_set = set(pd.to_datetime(feature_timestamps))
    
    # Find matches and non-matches
    matching_timestamps = duckdb_timestamps_set.intersection(calculated_timestamps_set)
    calculated_only = calculated_timestamps_set - duckdb_timestamps_set
    duckdb_only = duckdb_timestamps_set - calculated_timestamps_set
    
    print(f"\n✅ Results:")
    print(f"📊 Calculated timestamps: {len(calculated_timestamps_set)}")
    print(f"📊 DuckDB feature timestamps: {len(duckdb_timestamps_set)}")
    print(f"📊 Matching timestamps: {len(matching_timestamps)}")
    print(f"📊 DuckDB coverage: {len(matching_timestamps)}/{len(duckdb_timestamps_set)} ({len(matching_timestamps)/len(duckdb_timestamps_set)*100:.1f}%)")
    print(f"📊 Calculated coverage: {len(matching_timestamps)}/{len(calculated_timestamps_set)} ({len(matching_timestamps)/len(calculated_timestamps_set)*100:.1f}%)")
    
    # Show discrepancies summary only
    if len(calculated_only) > 0:
        print(f"\n❌ Calculated timestamps WITHOUT matching DuckDB features: {len(calculated_only)}")
    
    if len(duckdb_only) > 0:
        print(f"\n❌ DuckDB timestamps WITHOUT matching calculated features: {len(duckdb_only)}")
    
    # Analyze why calculated has different count than DuckDB features
    print(f"\n📊 COUNT ANALYSIS:")
    print(f"Calculated records: {len(calculated_features)}")
    print(f"DuckDB feature records: {len(feature_df)}")
    print(f"Difference: {len(feature_df) - len(calculated_features)} missing from calculated")
    
    # Find the exact missing timestamp by comparing timezone-naive timestamps
    if len(calculated_features) != len(feature_df):
        print(f"\n🔍 FINDING MISSING TIMESTAMP:")
        
        # One-to-one comparison using UTC timestamps
        calculated_timestamps_utc = [pd.to_datetime(record['timestamp']) for record in calculated_features]
        duckdb_timestamps_utc_list = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None).tolist()
        
        missing_in_calculated = set(duckdb_timestamps_utc_list) - set(calculated_timestamps_utc)
        missing_in_duckdb = set(calculated_timestamps_utc) - set(duckdb_timestamps_utc_list)
        
        if missing_in_calculated:
            print(f"  Missing from calculated: {len(missing_in_calculated)} timestamps")
            for missing_ts in sorted(missing_in_calculated):
                print(f"    {missing_ts}")
                
                # Find the DuckDB feature data for this timestamp
                feature_utc = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                matching_feature = feature_df[feature_utc == missing_ts]
                if len(matching_feature) > 0:
                    row = matching_feature.iloc[0]
                    print(f"      DuckDB: vol={row['total_volume_120s']:.6f}, buy_ratio={row['buy_ratio_60s']:.6f}")
                    print(f"      Original timestamp: {row['sample_timestamp']}")
                
        if missing_in_duckdb:
            print(f"  Missing from DuckDB: {len(missing_in_duckdb)} timestamps")
            for missing_ts in sorted(missing_in_duckdb):
                print(f"    {missing_ts}")
                
                # Find the calculated feature data for this timestamp
                matching_calculated = [r for r in calculated_features if pd.to_datetime(r['timestamp']) == missing_ts]
                if matching_calculated:
                    record = matching_calculated[0]
                    print(f"      Calculated: vol={record.get('total_volume_60s', 0):.6f}, buy_ratio={record.get('buy_ratio_60s', 0):.6f}")
    
    # 5. Feature Value Comparison
    print("\n" + "="*60)
    print("5. FEATURE VALUE COMPARISON")
    print("="*60)
    
    if len(calculated_features) > 0 and len(feature_df) > 0:
        print(f"Comparing calculated features vs DuckDB ground truth...")
        
        # Convert feature timestamps to UTC for matching
        feature_df_utc = feature_df.copy()
        feature_df_utc['sample_timestamp_utc'] = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Convert calculated timestamps to pandas datetime
        calculated_df = pd.DataFrame(calculated_features)
        calculated_df['timestamp_utc'] = pd.to_datetime(calculated_df['timestamp'])
        
        # Find matching timestamps and compare features
        feature_columns_to_compare = [
            'total_volume_30s', 'buy_ratio_30s', 'total_txns_30s', 'volume_concentration_30s', 'small_txn_ratio_30s',
            'total_volume_60s', 'buy_ratio_60s', 'total_txns_60s', 'volume_concentration_60s', 'small_txn_ratio_60s',
            'total_volume_120s', 'buy_ratio_120s', 'total_txns_120s', 'volume_concentration_120s', 
            'small_txn_ratio_120s', 'medium_txn_ratio_120s', 'whale_txn_ratio_120s'
        ]
        
        total_comparisons = 0
        exact_matches = 0
        close_matches = 0  # Within 1e-6 absolute or 1e-4 relative tolerance
        
        # Sample first 10 timestamps for detailed comparison
        sample_size = min(10, len(feature_df_utc))
        print(f"\nDetailed comparison for first {sample_size} timestamps:")
        print("-" * 60)
        
        for i in range(sample_size):
            feature_row = feature_df_utc.iloc[i]
            target_timestamp = feature_row['sample_timestamp_utc']
            
            # Find matching calculated record
            matching_calculated_records = calculated_df[calculated_df['timestamp_utc'] == target_timestamp]
            
            if len(matching_calculated_records) > 0:
                calculated_row = matching_calculated_records.iloc[0]
                
                print(f"\n📊 Timestamp: {target_timestamp}")
                print(f"   Original feature timestamp: {feature_row['sample_timestamp']}")
                
                timestamp_matches = 0
                timestamp_exact = 0
                timestamp_close = 0
                
                for feature_col in feature_columns_to_compare:
                    if feature_col in feature_row.index and feature_col in calculated_row.index:
                        ground_truth = feature_row[feature_col]
                        calculated = calculated_row[feature_col]
                        
                        # Compare with tolerance
                        if pd.isna(ground_truth) and pd.isna(calculated):
                            match_type = "EXACT"
                            is_exact = True
                            is_close = True
                        elif pd.isna(ground_truth) or pd.isna(calculated):
                            match_type = "MISMATCH"
                            is_exact = False
                            is_close = False
                        else:
                            diff = abs(float(ground_truth) - float(calculated))
                            is_exact = diff < 1e-10
                            is_close = diff < 1e-6 or (abs(diff / max(abs(float(ground_truth)), 1e-6)) < 1e-4)
                            
                            if is_exact:
                                match_type = "EXACT"
                            elif is_close:
                                match_type = "CLOSE"
                            else:
                                match_type = "MISMATCH"
                        
                        status = "✅" if is_close else "❌"
                        
                        # Only show mismatches and first few features to avoid clutter
                        if not is_close or feature_col.endswith('_60s'):  # Always show 60s features
                            print(f"   {status} {feature_col}: {ground_truth} vs {calculated} [{match_type}]")
                            if not is_close and not pd.isna(ground_truth) and not pd.isna(calculated):
                                diff = abs(float(ground_truth) - float(calculated))
                                rel_diff = abs(diff / max(abs(float(ground_truth)), 1e-6)) * 100
                                print(f"      Difference: {diff:.8f} (relative: {rel_diff:.4f}%)")
                        
                        timestamp_matches += 1
                        if is_exact:
                            timestamp_exact += 1
                        if is_close:
                            timestamp_close += 1
                        
                        total_comparisons += 1
                        if is_exact:
                            exact_matches += 1
                        if is_close:
                            close_matches += 1
                
                accuracy_exact = timestamp_exact / timestamp_matches * 100 if timestamp_matches > 0 else 0
                accuracy_close = timestamp_close / timestamp_matches * 100 if timestamp_matches > 0 else 0
                print(f"   📈 Timestamp accuracy: {timestamp_exact}/{timestamp_matches} exact ({accuracy_exact:.1f}%), {timestamp_close}/{timestamp_matches} close ({accuracy_close:.1f}%)")
            else:
                print(f"\n❌ No matching calculated record for {target_timestamp}")
        
        # Overall statistics
        print(f"\n" + "="*60)
        print("📊 OVERALL FEATURE COMPARISON STATISTICS")
        print("="*60)
        
        overall_exact_pct = exact_matches / total_comparisons * 100 if total_comparisons > 0 else 0
        overall_close_pct = close_matches / total_comparisons * 100 if total_comparisons > 0 else 0
        
        print(f"Total feature comparisons: {total_comparisons}")
        print(f"Exact matches: {exact_matches} ({overall_exact_pct:.1f}%)")
        print(f"Close matches (within tolerance): {close_matches} ({overall_close_pct:.1f}%)")
        print(f"Mismatches: {total_comparisons - close_matches} ({100 - overall_close_pct:.1f}%)")
        
        if overall_close_pct >= 99.0:
            print("\n🎉 Calculated features are highly accurate!")
        elif overall_close_pct >= 95.0:
            print("\n✅ Calculated features are mostly accurate")
        else:
            print("\n⚠️ Calculated features need investigation")
    else:
        print("❌ No data available for feature comparison")
    
    conn.close()

if __name__ == "__main__":
    main()