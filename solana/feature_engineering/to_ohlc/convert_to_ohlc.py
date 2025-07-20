#!/usr/bin/env python3
"""
Convert first_day_trades to OHLC format using SQL query
"""

import pandas as pd
import duckdb
import os
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DB_PATH = "/Volumes/Extreme SSD/DuckDB/solana.duckdb"
OUTPUT_DIR = "/Volumes/Extreme SSD/trading_data/solana/ohlc"
SQL_FILE = "solana/aggregate_to_ohlc.sql"

def create_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

def load_sql_query():
    """Load SQL query from file"""
    sql_file = Path(SQL_FILE)
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_FILE}")
    
    with open(sql_file, 'r') as f:
        sql_query = f.read()
    
    logging.info(f"Loaded SQL query from {SQL_FILE}")
    return sql_query

def convert_to_ohlc():
    """Convert first_day_trades to OHLC using SQL query"""
    logging.info("Starting OHLC conversion...")
    start_time = time.time()
    
    # Connect to database
    logging.info(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    
    # Check data availability
    count = conn.execute("SELECT COUNT(*) FROM first_day_trades").fetchone()[0]
    logging.info(f"Total records in first_day_trades: {count:,}")
    
    # Load and execute SQL query
    sql_query = load_sql_query()
    
    logging.info("Executing OHLC aggregation query...")
    ohlc_result = conn.execute(sql_query).fetchdf()
    
    execution_time = time.time() - start_time
    logging.info(f"✅ OHLC conversion completed in {execution_time:.2f}s")
    logging.info(f"Generated {len(ohlc_result):,} OHLC records")
    logging.info(f"Unique coins: {ohlc_result['mint'].nunique()}")
    
    conn.close()
    return ohlc_result

def save_ohlc_data(ohlc_df):
    """Save OHLC data to single file"""
    logging.info("Saving OHLC data...")
    
    # Save single combined file
    output_file = Path(OUTPUT_DIR) / "first_day_trades_ohlc.csv"
    ohlc_df.to_csv(output_file, index=False)
    logging.info(f"✅ Saved OHLC data: {output_file} ({len(ohlc_df):,} records)")
    
    return output_file


def show_sample_output(ohlc_df):
    """Show sample of the output"""
    print("\n" + "=" * 80)
    print("SAMPLE OHLC OUTPUT")
    print("=" * 80)
    
    sample = ohlc_df.head(3)
    for i, (idx, row) in enumerate(sample.iterrows(), 1):
        print(f"\nRecord {i}:")
        print(f"  Coin: {row['mint'][:12]}...")
        print(f"  Time: {row.get('time', 'N/A')}")
        print(f"  OHLC: O={row['open']:.8f}, H={row['high']:.8f}, L={row['low']:.8f}, C={row['close']:.8f}")
        print(f"  Volume: {row['sol_volume']:.4f} SOL ({row['transactions']} txns)")
    
    print(f"\nColumns: {list(ohlc_df.columns)}")
    print(f"Total records: {len(ohlc_df):,}")
    print(f"Date range: {ohlc_df['block_timestamp_iso'].min()} to {ohlc_df['block_timestamp_iso'].max()}")

def main():
    """Main conversion function"""
    print("=" * 80)
    print("CONVERTING FIRST_DAY_TRADES TO OHLC FORMAT")
    print("=" * 80)
    
    try:
        # Check database exists
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database not found: {DB_PATH}")
        
        # Create output directory
        create_output_directory()
        
        # Convert to OHLC
        ohlc_df = convert_to_ohlc()
        
        # Save results
        output_file = save_ohlc_data(ohlc_df)
        
        # Show sample
        show_sample_output(ohlc_df)
        
        print("\n" + "=" * 80)
        print("✅ CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"Output file: {output_file}")
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()