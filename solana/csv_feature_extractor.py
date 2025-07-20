#!/usr/bin/env python3
"""
CSV-based Feature Extraction for Solana Trading Data
Based on test_feature_extraction.py simple approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SOL_MINT = 'So11111111111111111111111111111111111111112'
SAMPLING_INTERVAL_SECONDS = 60
MIN_LOOKBACK_BUFFER_SECONDS = 120
FORWARD_PREDICTION_WINDOW_SECONDS = 300
MIN_TRANSACTIONS_PER_COIN = 100

def extract_features_single_coin_simple(df, coin_id, timestamp, window_seconds=30):
    """Extract features for a single coin at a single timestamp using simple logic."""
    coin_data = df[(df['mint'] == coin_id) & (df['succeeded'] == True)].copy()
    
    window_start = timestamp - timedelta(seconds=window_seconds)
    window_data = coin_data[
        (coin_data['block_timestamp'] >= window_start) & 
        (coin_data['block_timestamp'] < timestamp)
    ].copy()
    
    if len(window_data) == 0:
        return get_zero_features(window_seconds)
    
    # Add trading indicators
    window_data['is_buy'] = window_data['mint'] == window_data['swap_to_mint']
    window_data['is_sell'] = window_data['mint'] == window_data['swap_from_mint']
    
    # Calculate SOL amounts
    window_data['sol_amount'] = 0.0
    buy_mask = window_data['is_buy'] & (window_data['swap_from_mint'] == SOL_MINT)
    sell_mask = window_data['is_sell'] & (window_data['swap_to_mint'] == SOL_MINT)
    window_data.loc[buy_mask, 'sol_amount'] = window_data.loc[buy_mask, 'swap_from_amount']
    window_data.loc[sell_mask, 'sol_amount'] = window_data.loc[sell_mask, 'swap_to_amount']
    
    # Transaction size categories
    window_data['txn_size_category'] = 'Unknown'
    window_data.loc[window_data['sol_amount'] >= 100, 'txn_size_category'] = 'Whale'
    window_data.loc[(window_data['sol_amount'] >= 10) & (window_data['sol_amount'] < 100), 'txn_size_category'] = 'Big'
    window_data.loc[(window_data['sol_amount'] >= 1) & (window_data['sol_amount'] < 10), 'txn_size_category'] = 'Medium'
    window_data.loc[(window_data['sol_amount'] > 0) & (window_data['sol_amount'] < 1), 'txn_size_category'] = 'Small'
    
    # Calculate features
    total_volume = window_data['sol_amount'].sum()
    buy_volume = window_data[window_data['is_buy']]['sol_amount'].sum()
    sell_volume = window_data[window_data['is_sell']]['sol_amount'].sum()
    
    total_txns = len(window_data)
    buy_txns = window_data['is_buy'].sum()
    sell_txns = window_data['is_sell'].sum()
    
    unique_traders = window_data['swapper'].nunique()
    unique_buyers = window_data[window_data['is_buy']]['swapper'].nunique()
    unique_sellers = window_data[window_data['is_sell']]['swapper'].nunique()
    
    # Transaction size distribution
    size_dist = window_data['txn_size_category'].value_counts(normalize=True)
    
    features = {
        # Volume features
        f'total_volume_{window_seconds}s': total_volume,
        f'buy_volume_{window_seconds}s': buy_volume,
        f'sell_volume_{window_seconds}s': sell_volume,
        f'buy_ratio_{window_seconds}s': buy_volume / (total_volume + 1e-10),
        f'volume_imbalance_{window_seconds}s': (buy_volume - sell_volume) / (total_volume + 1e-10),
        
        # Transaction flow features
        f'total_txns_{window_seconds}s': total_txns,
        f'buy_txns_{window_seconds}s': buy_txns,
        f'sell_txns_{window_seconds}s': sell_txns,
        f'txn_buy_ratio_{window_seconds}s': buy_txns / (total_txns + 1e-10),
        f'txn_flow_imbalance_{window_seconds}s': (buy_txns - sell_txns) / (total_txns + 1e-10),
        
        # Trader behavior features
        f'unique_traders_{window_seconds}s': unique_traders,
        f'unique_buyers_{window_seconds}s': unique_buyers,
        f'unique_sellers_{window_seconds}s': unique_sellers,
        f'trader_buy_ratio_{window_seconds}s': unique_buyers / (unique_traders + 1e-10),
        
        # Transaction size features
        f'avg_txn_size_{window_seconds}s': total_volume / (total_txns + 1e-10),
        f'volume_per_trader_{window_seconds}s': total_volume / (unique_traders + 1e-10),
        f'volume_mean_{window_seconds}s': window_data['sol_amount'].mean() if total_volume > 0 else 0.0,
        f'volume_std_{window_seconds}s': np.std(window_data['sol_amount'], ddof=0) if total_volume > 0 else 0.0,
        f'volume_concentration_{window_seconds}s': np.std(window_data['sol_amount'], ddof=0) / window_data['sol_amount'].mean() if total_volume > 0 and window_data['sol_amount'].mean() > 0 else 0.0,
        
        # Size category ratios
        f'small_txn_ratio_{window_seconds}s': size_dist.get('Small', 0.0),
        f'medium_txn_ratio_{window_seconds}s': size_dist.get('Medium', 0.0),
        f'big_txn_ratio_{window_seconds}s': size_dist.get('Big', 0.0),
        f'whale_txn_ratio_{window_seconds}s': size_dist.get('Whale', 0.0),
    }
    
    return features

def get_zero_features(window_seconds):
    """Return zero features for empty windows."""
    return {
        f'total_volume_{window_seconds}s': 0.0,
        f'buy_volume_{window_seconds}s': 0.0,
        f'sell_volume_{window_seconds}s': 0.0,
        f'buy_ratio_{window_seconds}s': 0.0,
        f'volume_imbalance_{window_seconds}s': 0.0,
        f'total_txns_{window_seconds}s': 0,
        f'buy_txns_{window_seconds}s': 0,
        f'sell_txns_{window_seconds}s': 0,
        f'txn_buy_ratio_{window_seconds}s': 0.0,
        f'txn_flow_imbalance_{window_seconds}s': 0.0,
        f'unique_traders_{window_seconds}s': 0,
        f'unique_buyers_{window_seconds}s': 0,
        f'unique_sellers_{window_seconds}s': 0,
        f'trader_buy_ratio_{window_seconds}s': 0.0,
        f'avg_txn_size_{window_seconds}s': 0.0,
        f'volume_per_trader_{window_seconds}s': 0.0,
        f'volume_concentration_{window_seconds}s': 0.0,
        f'small_txn_ratio_{window_seconds}s': 0.0,
        f'medium_txn_ratio_{window_seconds}s': 0.0,
        f'big_txn_ratio_{window_seconds}s': 0.0,
        f'whale_txn_ratio_{window_seconds}s': 0.0,
        f'volume_mean_{window_seconds}s': 0.0,
        f'volume_std_{window_seconds}s': 0.0
    }

def process_single_csv(csv_file_path):
    """Process a single CSV file and extract features for all coins."""
    try:
        # Read CSV
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            logging.warning(f"Empty CSV: {os.path.basename(csv_file_path)}")
            return pd.DataFrame()
        
        df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
        df = df.sort_values('block_timestamp')
        
        # Filter to SOL-related trades
        sol_trades = df[
            ((df['swap_from_mint'] == SOL_MINT) | (df['swap_to_mint'] == SOL_MINT)) & 
            (df['mint'] != SOL_MINT) & 
            (df['succeeded'] == True)
        ].copy()
        
        if len(sol_trades) == 0:
            return pd.DataFrame()
        
        # Process each coin separately
        all_features = []
        coins = sol_trades['mint'].unique()
        
        for coin_id in coins:
            import time
            coin_start_time = time.time()
            
            coin_data = sol_trades[sol_trades['mint'] == coin_id].copy()
            
            if len(coin_data) < MIN_TRANSACTIONS_PER_COIN:
                logging.info(f"Skipping {coin_id[:8]}... - only {len(coin_data)} transactions (need {MIN_TRANSACTIONS_PER_COIN})")
                continue
            
            # Define valid sampling window
            start_time = coin_data['block_timestamp'].min() + timedelta(seconds=MIN_LOOKBACK_BUFFER_SECONDS)
            end_time = coin_data['block_timestamp'].max() - timedelta(seconds=FORWARD_PREDICTION_WINDOW_SECONDS)
            
            if start_time >= end_time:
                logging.info(f"Skipping {coin_id[:8]}... - insufficient time range")
                continue
            
            # Generate sample timestamps
            sample_times = pd.date_range(start_time, end_time, freq=f'{SAMPLING_INTERVAL_SECONDS}s')
            
            logging.info(f"Processing {coin_id[:8]}... - {len(coin_data)} txns, {len(sample_times)} samples")
            
            coin_features = []
            for timestamp in sample_times:
                result = {'coin_id': coin_id, 'sample_timestamp': timestamp}
                
                # Extract features for all three windows
                for window in [30, 60, 120]:
                    features = extract_features_single_coin_simple(sol_trades, coin_id, timestamp, window)
                    result.update(features)
                
                # Add forward profitability
                forward_start = timestamp
                forward_end = timestamp + timedelta(seconds=FORWARD_PREDICTION_WINDOW_SECONDS)
                forward_data = coin_data[(coin_data['block_timestamp'] >= forward_start) & (coin_data['block_timestamp'] < forward_end)].copy()
                
                if len(forward_data) > 0:
                    forward_data['is_buy'] = forward_data['mint'] == forward_data['swap_to_mint']
                    forward_data['is_sell'] = forward_data['mint'] == forward_data['swap_from_mint']
                    forward_data['sol_amount'] = 0.0
                    buy_mask = forward_data['is_buy'] & (forward_data['swap_from_mint'] == SOL_MINT)
                    sell_mask = forward_data['is_sell'] & (forward_data['swap_to_mint'] == SOL_MINT)
                    forward_data.loc[buy_mask, 'sol_amount'] = forward_data.loc[buy_mask, 'swap_from_amount']
                    forward_data.loc[sell_mask, 'sol_amount'] = forward_data.loc[sell_mask, 'swap_to_amount']
                    
                    forward_buy_volume = forward_data[forward_data['is_buy']]['sol_amount'].sum()
                    forward_sell_volume = forward_data[forward_data['is_sell']]['sol_amount'].sum()
                    result['forward_buy_volume_300s'] = forward_buy_volume
                    result['forward_sell_volume_300s'] = forward_sell_volume
                    result['is_profitable_300s'] = 1 if forward_buy_volume > forward_sell_volume else 0
                else:
                    result['forward_buy_volume_300s'] = 0.0
                    result['forward_sell_volume_300s'] = 0.0
                    result['is_profitable_300s'] = 0
                
                coin_features.append(result)
            
            # Log completion for this coin
            coin_end_time = time.time()
            coin_duration = coin_end_time - coin_start_time
            logging.info(f"✅ {coin_id[:8]}... completed - {len(coin_features)} features in {coin_duration:.2f}s")
            all_features.extend(coin_features)
        
        if all_features:
            return pd.DataFrame(all_features)
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(csv_file_path)}: {e}")
        return pd.DataFrame()

def process_csv_batch(csv_files, output_dir, batch_id):
    """Process a batch of CSV files."""
    all_features = []
    
    for csv_file in csv_files:
        logging.info(f"Batch {batch_id}: Processing {os.path.basename(csv_file)}")
        features_df = process_single_csv(csv_file)
        if not features_df.empty:
            all_features.append(features_df)
    
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        output_file = os.path.join(output_dir, f'features_batch_{batch_id}.csv')
        combined_features.to_csv(output_file, index=False)
        logging.info(f"Batch {batch_id}: Saved {len(combined_features)} features to {output_file}")
        return output_file
    else:
        logging.warning(f"Batch {batch_id}: No features extracted")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract features from Solana CSV files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save feature files')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of CSV files per batch')
    parser.add_argument('--max_workers', type=int, default=None, help='Max parallel workers')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find CSV files
    csv_pattern = os.path.join(args.input_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('._')]
    
    logging.info(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        logging.error("No CSV files found!")
        return
    
    # Split into batches
    batches = [csv_files[i:i+args.batch_size] for i in range(0, len(csv_files), args.batch_size)]
    
    max_workers = args.max_workers or min(multiprocessing.cpu_count(), len(batches))
    logging.info(f"Processing {len(batches)} batches with {max_workers} workers")
    
    completed_batches = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_csv_batch, batch, args.output_dir, i): i 
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                result = future.result()
                completed_batches += 1
                logging.info(f"Completed batch {batch_id+1}/{len(batches)}")
            except Exception as e:
                logging.error(f"Batch {batch_id} failed: {e}")
    
    # Combine results
    logging.info("Combining all results...")
    feature_files = glob.glob(os.path.join(args.output_dir, "features_batch_*.csv"))
    
    if feature_files:
        all_features = []
        for feature_file in feature_files:
            df = pd.read_csv(feature_file)
            all_features.append(df)
        
        combined_df = pd.concat(all_features, ignore_index=True)
        final_output = os.path.join(args.output_dir, "combined_features.csv")
        combined_df.to_csv(final_output, index=False)
        
        logging.info(f"✅ Final features: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
        
        # Cleanup
        for feature_file in feature_files:
            os.remove(feature_file)
    else:
        logging.error("No feature files generated!")

if __name__ == '__main__':
    main()