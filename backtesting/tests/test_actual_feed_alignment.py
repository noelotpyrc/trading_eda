#!/usr/bin/env python3
"""
Test actual SolanaTransactionFeed alignment with feature data
Uses the real feed implementation instead of simulating timestamp logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import duckdb
import backtrader as bt
from datetime import datetime, timedelta
from solana_transaction_feed import SolanaTransactionFeed

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
        END as sol_amount
    FROM first_day_trades
    WHERE mint = '{coin_id}'
    AND succeeded = TRUE
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
    
    # Create actual SolanaTransactionFeed and run it
    print("\n3. Running actual SolanaTransactionFeed...")
    
    # Create a minimal Cerebro just to run the feed
    cerebro = bt.Cerebro()
    
    # Create the actual feed with explicit feature boundaries
    feature_start = feature_df['sample_timestamp'].min()
    feature_end = feature_df['sample_timestamp'].max()
    
    print(f"  Using feature boundaries: {feature_start} to {feature_end}")
    
    data_feed = SolanaTransactionFeed(
        transaction_data=transaction_df,
        aggregation_window=60,
        min_lookback_seconds=120,
        min_forward_seconds=300,
        feature_start_time=feature_start,
        feature_end_time=feature_end,
        datetime_col='block_timestamp',
        volume_col='sol_amount',
        direction_col='is_buy',
        trader_col='swapper',
        succeed_col='succeeded'
    )
    
    # Add feed to cerebro
    cerebro.adddata(data_feed)
    
    # Capture feed timestamps and transaction data by running a simple strategy
    feed_data = []
    
    class TimestampCapture(bt.Strategy):
        def next(self):
            current_datetime = self.datas[0].datetime.datetime(0)
            
            # Capture feed data including transaction metrics and ML features
            feed_record = {
                'timestamp': current_datetime,
                'volume': self.data.volume[0],
                'transaction_count': self.data.transaction_count[0],
                'buy_ratio': self.data.is_buy[0],
                'total_volume': self.data.sol_amount[0],
                
                # ML Features from WindowFeatureCalculator
                'total_volume_30s': self.data.total_volume_30s[0],
                'buy_ratio_30s': self.data.buy_ratio_30s[0],
                'total_txns_30s': self.data.total_txns_30s[0],
                'volume_concentration_30s': self.data.volume_concentration_30s[0],
                'small_txn_ratio_30s': self.data.small_txn_ratio_30s[0],
                
                'total_volume_60s': self.data.total_volume_60s[0],
                'buy_ratio_60s': self.data.buy_ratio_60s[0],
                'total_txns_60s': self.data.total_txns_60s[0],
                'volume_concentration_60s': self.data.volume_concentration_60s[0],
                'small_txn_ratio_60s': self.data.small_txn_ratio_60s[0],
                
                'total_volume_120s': self.data.total_volume_120s[0],
                'buy_ratio_120s': self.data.buy_ratio_120s[0],
                'total_txns_120s': self.data.total_txns_120s[0],
                'volume_concentration_120s': self.data.volume_concentration_120s[0],
                'small_txn_ratio_120s': self.data.small_txn_ratio_120s[0],
                'medium_txn_ratio_120s': self.data.medium_txn_ratio_120s[0],
                'whale_txn_ratio_120s': self.data.whale_txn_ratio_120s[0]
            }
            feed_data.append(feed_record)
            
            # Debug first few timestamps with ML features
            if len(feed_data) <= 3:
                print(f"  Debug {len(feed_data)}: {current_datetime}")
                print(f"    Volume: {feed_record['volume']:.6f} | Txns: {feed_record['transaction_count']} | Buy ratio: {feed_record['buy_ratio']:.3f}")
                print(f"    ML Features - vol_60s: {feed_record['total_volume_60s']:.6f} | buy_ratio_60s: {feed_record['buy_ratio_60s']:.3f} | txns_60s: {feed_record['total_txns_60s']}")
            
            # Print progress every 100 timestamps
            if len(feed_data) % 100 == 0:
                print(f"  Captured {len(feed_data)} feed records...")
    
    cerebro.addstrategy(TimestampCapture)
    
    # Run the feed
    print("  Running feed to capture actual timestamps...")
    results = cerebro.run()
    
    # Extract timestamps from feed data
    feed_timestamps = [record['timestamp'] for record in feed_data]
    
    print(f"✅ Feed generated {len(feed_timestamps)} timestamps")
    if len(feed_timestamps) > 0:
        print(f"Feed time range: {min(feed_timestamps)} to {max(feed_timestamps)}")
    
    # Compare with features
    print("\n4. Comparing actual feed timestamps with features...")
    
    if len(feature_df) == 0 or len(feed_timestamps) == 0:
        print("❌ No data to compare")
        return
    
    # Convert feature timestamps to UTC to match feed timestamps
    feature_timestamps_utc = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    feature_timestamps = set(feature_timestamps_utc)
    feed_timestamps_set = set(pd.to_datetime(feed_timestamps))
    
    # Find matches and non-matches
    matching_timestamps = feature_timestamps.intersection(feed_timestamps_set)
    feed_only = feed_timestamps_set - feature_timestamps
    feature_only = feature_timestamps - feed_timestamps_set
    
    print(f"\n✅ Results:")
    print(f"📊 Feed timestamps: {len(feed_timestamps_set)}")
    print(f"📊 Feature timestamps: {len(feature_timestamps)}")
    print(f"📊 Matching timestamps: {len(matching_timestamps)}")
    print(f"📊 Feature coverage: {len(matching_timestamps)}/{len(feature_timestamps)} ({len(matching_timestamps)/len(feature_timestamps)*100:.1f}%)")
    print(f"📊 Feed coverage: {len(matching_timestamps)}/{len(feed_timestamps_set)} ({len(matching_timestamps)/len(feed_timestamps_set)*100:.1f}%)")
    
    # Show discrepancies summary only
    if len(feed_only) > 0:
        print(f"\n❌ Feed timestamps WITHOUT matching features: {len(feed_only)}")
    
    if len(feature_only) > 0:
        print(f"\n❌ Feature timestamps WITHOUT matching feed: {len(feature_only)}")
    
    # Analyze why feed has different count than features
    print(f"\n📊 COUNT ANALYSIS:")
    print(f"Feed records: {len(feed_data)}")
    print(f"Feature records: {len(feature_df)}")
    print(f"Difference: {len(feature_df) - len(feed_data)} missing from feed")
    
    # Find the exact missing timestamp by comparing timezone-naive timestamps
    if len(feed_data) != len(feature_df):
        print(f"\n🔍 FINDING MISSING TIMESTAMP:")
        
        # One-to-one comparison using UTC timestamps
        feed_timestamps_utc = [pd.to_datetime(record['timestamp']) for record in feed_data]
        feature_timestamps_utc_list = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None).tolist()
        
        missing_in_feed = set(feature_timestamps_utc_list) - set(feed_timestamps_utc)
        missing_in_features = set(feed_timestamps_utc) - set(feature_timestamps_utc_list)
        
        if missing_in_feed:
            print(f"  Missing from feed: {len(missing_in_feed)} timestamps")
            for missing_ts in sorted(missing_in_feed):
                print(f"    {missing_ts}")
                
                # Find the feature data for this timestamp
                feature_utc = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                matching_feature = feature_df[feature_utc == missing_ts]
                if len(matching_feature) > 0:
                    row = matching_feature.iloc[0]
                    print(f"      Feature: vol={row['total_volume_120s']:.6f}, buy_ratio={row['buy_ratio_60s']:.6f}")
                    print(f"      Original timestamp: {row['sample_timestamp']}")
                
        if missing_in_features:
            print(f"  Missing from features: {len(missing_in_features)} timestamps")
            for missing_ts in sorted(missing_in_features):
                print(f"    {missing_ts}")
                
                # Find the feed data for this timestamp
                matching_feed = [r for r in feed_data if pd.to_datetime(r['timestamp']) == missing_ts]
                if matching_feed:
                    record = matching_feed[0]
                    print(f"      Feed: vol={record['total_volume']:.6f}, buy_ratio={record['buy_ratio']:.6f}")
    
    # 5. Feature Value Comparison
    print("\n" + "="*60)
    print("5. FEATURE VALUE COMPARISON")
    print("="*60)
    
    if len(feed_data) > 0 and len(feature_df) > 0:
        print(f"Comparing ML features from feed vs DuckDB ground truth...")
        
        # Convert feature timestamps to UTC for matching
        feature_df_utc = feature_df.copy()
        feature_df_utc['sample_timestamp_utc'] = feature_df['sample_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Convert feed timestamps to pandas datetime
        feed_df = pd.DataFrame(feed_data)
        feed_df['timestamp_utc'] = pd.to_datetime(feed_df['timestamp'])
        
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
            
            # Find matching feed record
            matching_feed_records = feed_df[feed_df['timestamp_utc'] == target_timestamp]
            
            if len(matching_feed_records) > 0:
                feed_row = matching_feed_records.iloc[0]
                
                print(f"\n📊 Timestamp: {target_timestamp}")
                print(f"   Original feature timestamp: {feature_row['sample_timestamp']}")
                
                timestamp_matches = 0
                timestamp_exact = 0
                timestamp_close = 0
                
                for feature_col in feature_columns_to_compare:
                    if feature_col in feature_row.index and feature_col in feed_row.index:
                        ground_truth = feature_row[feature_col]
                        calculated = feed_row[feature_col]
                        
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
                print(f"\n❌ No matching feed record for {target_timestamp}")
        
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
            print("\n🎉 Feed features are highly accurate!")
        elif overall_close_pct >= 95.0:
            print("\n✅ Feed features are mostly accurate")
        else:
            print("\n⚠️ Feed features need investigation")
    else:
        print("❌ No data available for feature comparison")
    
    conn.close()

if __name__ == "__main__":
    main()