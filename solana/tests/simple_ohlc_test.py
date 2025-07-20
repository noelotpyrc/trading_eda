#!/usr/bin/env python3
"""
Simple test to compare SQL vs Python OHLC results using existing DuckDB data
"""

import pandas as pd
import numpy as np
import duckdb
import sys
sys.path.append('.')

# Database path
db_path = "/Volumes/Extreme SSD/DuckDB/solana.duckdb"

# Test with specific coin (using one coin from batch 578)
test_coin = "4kgcTW3fy28KC659Hqwvpwvsk9zRH88oDPYPnYrnefZr"

print("=== TESTING OHLC IMPLEMENTATIONS ===\n")

# 1. Run SQL version
print("1. Running SQL version...")
conn = duckdb.connect(db_path)

# Read SQL file and add coin filter
with open("solana/aggregate_to_ohlc.sql", 'r') as f:
    sql_query = f.read()

# Add WHERE clause to filter for specific coin in clean_trades CTE
sql_query = sql_query.replace(
    "AND t.mint != c.SOL_MINT",
    f"AND t.mint != c.SOL_MINT\n        AND t.mint = '{test_coin}'"
)

# Execute SQL
sql_result = conn.execute(sql_query).fetchdf()
conn.close()

print(f"SQL result: {len(sql_result)} records")

# 2. Run Python version with same coin
print("\n2. Running Python version...")

# Connect to get the same coin data
conn = duckdb.connect(db_path)
test_data = conn.execute(f"SELECT * FROM first_day_trades WHERE mint = '{test_coin}'").fetchdf()
conn.close()

print(f"Loaded {len(test_data)} transactions for test coin")

# Save to temp CSV for Python script
test_data.to_csv("/tmp/test_coin_data.csv", index=False)

# Run Python aggregation
from solana.aggregate_to_ohlc import load_solana_data, prepare_trading_data, aggregate_to_ohlc, format_output_columns

df = load_solana_data("/tmp/test_coin_data.csv")
df_clean = prepare_trading_data(df)
ohlc_df = aggregate_to_ohlc(df_clean)
python_result = format_output_columns(ohlc_df)

print(f"Python result: {len(python_result)} records")

# 3. Compare results
print("\n3. Comparing results...")

if len(sql_result) != len(python_result):
    print(f"❌ Different record counts: SQL={len(sql_result)}, Python={len(python_result)}")
else:
    print(f"✅ Same record count: {len(sql_result)}")

# Prepare for comparison
sql_df = sql_result.copy()
python_df = python_result.copy()

# Sort both by timestamp for comparison
if 'block_timestamp_iso' in sql_df.columns:
    sql_df['timestamp'] = pd.to_datetime(sql_df['block_timestamp_iso'])
else:
    sql_df['timestamp'] = pd.to_datetime(sql_df['block_timestamp'])

python_df['timestamp'] = pd.to_datetime(python_df['block_timestamp_iso'])

sql_df = sql_df.sort_values('timestamp').reset_index(drop=True)
python_df = python_df.sort_values('timestamp').reset_index(drop=True)

# Compare if same length
if len(sql_df) == len(python_df):
    print("\n4. OHLC Comparison:")
    ohlc_cols = ['open', 'high', 'low', 'close']
    
    for col in ohlc_cols:
        sql_vals = sql_df[col].values
        python_vals = python_df[col].values
        
        diff = np.abs(sql_vals - python_vals)
        max_diff = diff.max()
        mean_diff = diff.mean()
        exact_matches = (diff < 1e-10).sum()
        
        print(f"{col.upper()}: {exact_matches}/{len(sql_df)} exact matches, max diff: {max_diff:.2e}")

    print("\n5. Volume Comparison:")
    vol_cols = ['sol_volume', 'vwap']
    
    for col in vol_cols:
        if col in sql_df.columns and col in python_df.columns:
            sql_vals = sql_df[col].values
            python_vals = python_df[col].values
            
            rel_diff = np.abs((sql_vals - python_vals) / (sql_vals + 1e-18))
            max_rel_diff = rel_diff.max()
            close_matches = (rel_diff < 0.01).sum()
            
            print(f"{col.upper()}: {close_matches}/{len(sql_df)} within 1%, max rel diff: {max_rel_diff:.2%}")
            
            # Show the record with max difference
            if max_rel_diff > 0.01:  # If there's a difference > 1%
                max_diff_idx = rel_diff.argmax()
                print(f"  Record with max difference (index {max_diff_idx}):")
                print(f"    Timestamp: {sql_df.iloc[max_diff_idx]['timestamp']}")
                print(f"    SQL {col}: {sql_vals[max_diff_idx]:.8f}")
                print(f"    Python {col}: {python_vals[max_diff_idx]:.8f}")
                print(f"    Relative diff: {rel_diff[max_diff_idx]:.2%}")

    # Show first few records
    print(f"\n6. First 3 records comparison:")
    for i in range(min(3, len(sql_df))):
        print(f"\nRecord {i+1}:")
        print(f"  Timestamp: {sql_df.iloc[i]['timestamp']}")
        print(f"  OPEN:  SQL={sql_df.iloc[i]['open']:.8f}, Python={python_df.iloc[i]['open']:.8f}")
        print(f"  HIGH:  SQL={sql_df.iloc[i]['high']:.8f}, Python={python_df.iloc[i]['high']:.8f}")
        print(f"  LOW:   SQL={sql_df.iloc[i]['low']:.8f}, Python={python_df.iloc[i]['low']:.8f}")
        print(f"  CLOSE: SQL={sql_df.iloc[i]['close']:.8f}, Python={python_df.iloc[i]['close']:.8f}")
        if 'sol_volume' in sql_df.columns:
            print(f"  VOL:   SQL={sql_df.iloc[i]['sol_volume']:.4f}, Python={python_df.iloc[i]['sol_volume']:.4f}")

else:
    print("❌ Cannot compare - different number of records")
    print(f"SQL timestamps: {sql_df['timestamp'].min()} to {sql_df['timestamp'].max()}")
    print(f"Python timestamps: {python_df['timestamp'].min()} to {python_df['timestamp'].max()}")

print("\n✅ Test completed!")