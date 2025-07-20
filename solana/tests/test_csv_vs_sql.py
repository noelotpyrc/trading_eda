#!/usr/bin/env python3
"""
Test script to compare CSV feature extractor vs SQL batch extraction
on the same batch 578 CSV file
"""

import pandas as pd
import numpy as np
import duckdb
import tempfile
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append('.')
from solana.feature_engineering.csv_feature_extractor import process_single_csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_batch_578_data():
    """Load the batch 578 CSV file"""
    print("Loading batch 578 data...")
    
    # Try different possible paths
    possible_paths = [
        "data/solana/first_day_trades/first_day_trades_batch_578.csv",
        "../data/solana/first_day_trades/first_day_trades_batch_578.csv"
    ]
    
    for data_path in possible_paths:
        if Path(data_path).exists():
            df = pd.read_csv(data_path)
            df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
            print(f"✅ Loaded {len(df):,} transactions from {data_path}")
            print(f"   Unique coins: {df['mint'].nunique()}")
            print(f"   Date range: {df['block_timestamp'].min()} to {df['block_timestamp'].max()}")
            return df
    
    print(f"❌ Data file not found in any of these paths:")
    for path in possible_paths:
        print(f"   {path}")
    return None

def run_csv_extractor(df):
    """Run CSV feature extractor on the data"""
    print(f"\n=== RUNNING CSV FEATURE EXTRACTOR ===")
    
    # Save DataFrame to temporary CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        temp_csv_path = tmp.name
    
    try:
        start_time = time.time()
        
        print(f"Processing temporary CSV: {temp_csv_path}")
        csv_results = process_single_csv(temp_csv_path)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if csv_results.empty:
            print("❌ CSV extractor returned empty results")
            return None
        
        print(f"✅ CSV extractor completed in {duration:.2f}s")
        print(f"   Generated {len(csv_results):,} feature samples")
        print(f"   {len(csv_results.columns)} columns")
        print(f"   Unique coins: {csv_results['coin_id'].nunique()}")
        
        return csv_results
        
    finally:
        # Cleanup temp file
        os.unlink(temp_csv_path)

def run_sql_batch_extractor(df):
    """Run SQL batch feature extractor on the data"""
    print(f"\n=== RUNNING SQL BATCH EXTRACTOR ===")
    
    try:
        # Connect to existing DuckDB file
        db_path = "/Volumes/Extreme SSD/DuckDB/solana.duckdb"
        print(f"Connecting to existing DuckDB: {db_path}")
        conn = duckdb.connect(db_path)
        
        # Read the SQL file
        sql_file = Path("solana/batch_feature_extraction.sql")
        if not sql_file.exists():
            print(f"❌ SQL file not found: {sql_file}")
            return None
        
        with open(sql_file, 'r') as f:
            full_sql = f.read()
        
        # Get all unique coins from the CSV data for filtering
        SOL_MINT = 'So11111111111111111111111111111111111111112'
        sol_trades = df[
            ((df['swap_from_mint'] == SOL_MINT) | (df['swap_to_mint'] == SOL_MINT)) & 
            (df['mint'] != SOL_MINT) & 
            (df['succeeded'] == True)
        ]
        csv_coins = sol_trades['mint'].unique().tolist()
        
        print(f"Filtering SQL query for {len(csv_coins)} coins from the CSV file")
        
        # Check what's in the DuckDB
        tables = conn.execute("SHOW TABLES").fetchdf()
        print(f"Available tables in DuckDB: {tables['name'].tolist()}")
        
        # Check if first_day_trades table exists
        if 'first_day_trades' not in tables['name'].values:
            print("❌ 'first_day_trades' table not found in DuckDB")
            return None
        
        # Modify SQL to filter for these specific coins from the CSV
        csv_coins_str = "', '".join(csv_coins)
        modified_sql = full_sql.replace(
            "AND t.succeeded = TRUE\n    GROUP",
            f"AND t.succeeded = TRUE AND t.mint IN ('{csv_coins_str}')\n    GROUP"
        )
        modified_sql = modified_sql.replace(
            "AND t.succeeded = TRUE\n),",
            f"AND t.succeeded = TRUE AND t.mint IN ('{csv_coins_str}')\n),",
            1  # Only replace the second occurrence
        )
        
        start_time = time.time()
        
        print("Executing SQL batch query...")
        sql_results = conn.execute(modified_sql).fetchdf()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ SQL batch completed in {duration:.2f}s")
        print(f"   Generated {len(sql_results):,} feature samples")
        print(f"   {len(sql_results.columns)} columns")
        print(f"   Unique coins: {sql_results['coin_id'].nunique()}")
        
        conn.close()
        return sql_results
        
    except Exception as e:
        print(f"❌ SQL batch extractor failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(csv_results, sql_results):
    """Compare results from both approaches"""
    print(f"\n=== COMPARING RESULTS ===")
    
    if csv_results is None or sql_results is None:
        print("❌ Cannot compare - one or both approaches failed")
        return
    
    print(f"CSV approach: {len(csv_results):,} samples, {csv_results['coin_id'].nunique()} coins")
    print(f"SQL approach: {len(sql_results):,} samples, {sql_results['coin_id'].nunique()} coins")
    
    # Compare coin coverage
    csv_coins = set(csv_results['coin_id'].unique())
    sql_coins = set(sql_results['coin_id'].unique())
    common_coins = csv_coins.intersection(sql_coins)
    
    print(f"\nCoin coverage:")
    print(f"  Common coins: {len(common_coins)}")
    print(f"  CSV only: {len(csv_coins - sql_coins)}")
    print(f"  SQL only: {len(sql_coins - csv_coins)}")
    
    if len(common_coins) == 0:
        print("❌ No common coins to compare!")
        return
    
    # Filter to common coins for detailed comparison
    csv_common = csv_results[csv_results['coin_id'].isin(common_coins)].copy()
    sql_common = sql_results[sql_results['coin_id'].isin(common_coins)].copy()
    
    # Sort both for comparison
    csv_common = csv_common.sort_values(['coin_id', 'sample_timestamp']).reset_index(drop=True)
    sql_common = sql_common.sort_values(['coin_id', 'sample_timestamp']).reset_index(drop=True)
    
    print(f"\nAfter filtering to common coins:")
    print(f"  CSV: {len(csv_common):,} samples")
    print(f"  SQL: {len(sql_common):,} samples")
    
    # Compare column structures
    csv_cols = set(csv_common.columns)
    sql_cols = set(sql_common.columns)
    common_cols = csv_cols.intersection(sql_cols)
    
    print(f"\nColumn comparison:")
    print(f"  Common columns: {len(common_cols)}")
    print(f"  CSV only: {csv_cols - sql_cols}")
    print(f"  SQL only: {sql_cols - common_cols}")
    
    # Sample comparison for a few coins
    print(f"\n=== SAMPLE COMPARISON ===")
    sample_coins = list(common_coins)[:3]  # Take first 3 coins
    
    for coin in sample_coins:
        csv_coin = csv_common[csv_common['coin_id'] == coin].head(2)
        sql_coin = sql_common[sql_common['coin_id'] == coin].head(2)
        
        print(f"\nCoin {coin[:8]}...")
        print(f"  CSV samples: {len(csv_coin)}")
        print(f"  SQL samples: {len(sql_coin)}")
        
        if len(csv_coin) > 0 and len(sql_coin) > 0:
            # Compare a few key features
            key_features = ['total_volume_60s', 'buy_ratio_60s', 'unique_traders_60s']
            for feature in key_features:
                if feature in csv_coin.columns and feature in sql_coin.columns:
                    csv_val = csv_coin[feature].iloc[0]
                    sql_val = sql_coin[feature].iloc[0]
                    diff = abs(csv_val - sql_val) if pd.notna(csv_val) and pd.notna(sql_val) else 'N/A'
                    print(f"    {feature}: CSV={csv_val:.4f}, SQL={sql_val:.4f}, diff={diff}")

def main():
    """Main test function"""
    print("="*70)
    print("TESTING CSV FEATURE EXTRACTOR vs SQL BATCH EXTRACTOR")
    print("="*70)
    
    # Load data
    df = load_batch_578_data()
    if df is None:
        return
    
    # Run both approaches
    csv_results = run_csv_extractor(df)
    sql_results = run_sql_batch_extractor(df)
    
    # Compare results
    compare_results(csv_results, sql_results)
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    main()