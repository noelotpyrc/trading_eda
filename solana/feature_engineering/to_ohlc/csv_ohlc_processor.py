#!/usr/bin/env python3
"""
CSV-based OHLC processor for Solana trading data
Reads CSV files to get unique coins, then uses those as filters 
in SQL query against first_day_trades table to generate OHLC data
"""

import pandas as pd
import duckdb
import os
import glob
import logging
import time
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_csv_to_ohlc(csv_file_path, sql_query, output_dir, db_path):
    """Process a single CSV file by getting its coins and filtering OHLC query"""
    try:
        csv_name = os.path.basename(csv_file_path)
        logging.info(f"Processing {csv_name}")
        
        start_time = time.time()
        
        # Read CSV to get unique coins
        df = pd.read_csv(csv_file_path)
        unique_coins = df['mint'].unique().tolist()
        
        if len(unique_coins) == 0:
            logging.warning(f"No coins found in {csv_name}")
            return None
        
        logging.info(f"{csv_name}: {len(df):,} transactions, {len(unique_coins)} unique coins")
        
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # Create WHERE clause for coin filtering
        coin_list = "', '".join(unique_coins)
        where_clause = f"AND t.mint IN ('{coin_list}')"
        
        # Modify SQL query to add coin filter
        modified_sql = sql_query.replace(
            "AND t.mint != c.SOL_MINT",
            f"AND t.mint != c.SOL_MINT\n        {where_clause}"
        )
        
        # Execute OHLC aggregation SQL
        try:
            ohlc_df = conn.execute(modified_sql).fetchdf()
            conn.close()
            
            if ohlc_df.empty:
                logging.warning(f"No OHLC data extracted from {csv_name}")
                return None
            
            # Save results
            output_file = os.path.join(output_dir, f"ohlc_{csv_name}")
            ohlc_df.to_csv(output_file, index=False)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logging.info(f"✅ {csv_name}: {len(ohlc_df):,} OHLC records in {duration:.2f}s -> {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"SQL execution failed for {csv_name}: {e}")
            conn.close()
            return None
            
    except Exception as e:
        logging.error(f"Error processing {csv_file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Process CSV files using SQL OHLC aggregation')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save OHLC files')
    parser.add_argument('--db_path', type=str, default='/Volumes/Extreme SSD/DuckDB/solana.duckdb', help='DuckDB file path')
    parser.add_argument('--sql_file', type=str, default='solana/aggregate_to_ohlc.sql', help='SQL file path')
    parser.add_argument('--combine_output', action='store_true', help='Combine all results into single file')
    parser.add_argument('--final_output', type=str, default='first_day_trades_ohlc.csv', help='Final combined output filename')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read SQL query
    sql_file = Path(args.sql_file)
    if not sql_file.exists():
        logging.error(f"SQL file not found: {sql_file}")
        return
    
    with open(sql_file, 'r') as f:
        sql_query = f.read()
    
    logging.info(f"Loaded SQL query from {sql_file}")
    
    # Find CSV files
    csv_pattern = os.path.join(args.input_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith('._')]
    
    logging.info(f"Found {len(csv_files)} CSV files to process")
    
    if not csv_files:
        logging.error("No CSV files found!")
        return
    
    # Verify DuckDB exists
    if not os.path.exists(args.db_path):
        logging.error(f"DuckDB file not found: {args.db_path}")
        return
    
    logging.info(f"Using DuckDB: {args.db_path}")
    
    # Process files sequentially
    all_output_files = []
    start_time = time.time()
    
    for i, csv_file in enumerate(csv_files, 1):
        logging.info(f"Processing file {i}/{len(csv_files)}")
        result = process_single_csv_to_ohlc(csv_file, sql_query, args.output_dir, args.db_path)
        if result:
            all_output_files.append(result)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    logging.info(f"Processing completed in {total_duration:.2f}s")
    logging.info(f"Successfully processed {len(all_output_files)}/{len(csv_files)} CSV files")
    
    # Combine results if requested
    if args.combine_output and all_output_files:
        logging.info("Combining all OHLC files...")
        
        all_ohlc_data = []
        for ohlc_file in all_output_files:
            try:
                df = pd.read_csv(ohlc_file)
                all_ohlc_data.append(df)
            except Exception as e:
                logging.error(f"Error reading {ohlc_file}: {e}")
        
        if all_ohlc_data:
            combined_df = pd.concat(all_ohlc_data, ignore_index=True)
            
            # Sort by mint and timestamp
            combined_df = combined_df.sort_values(['mint', 'block_timestamp_iso']).reset_index(drop=True)
            
            final_output = os.path.join(args.output_dir, args.final_output)
            combined_df.to_csv(final_output, index=False)
            
            logging.info(f"✅ Combined OHLC data saved: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
            logging.info(f"Final output: {final_output}")
            logging.info(f"Unique coins: {combined_df['mint'].nunique()}")
            logging.info(f"Date range: {combined_df['block_timestamp_iso'].min()} to {combined_df['block_timestamp_iso'].max()}")
            
            # Show sample statistics
            logging.info(f"Total SOL volume: {combined_df['sol_volume'].sum():,.2f}")
            logging.info(f"Total transactions: {combined_df['transactions'].sum():,}")
            
            # Cleanup individual files
            for ohlc_file in all_output_files:
                os.remove(ohlc_file)
            logging.info("Cleaned up individual OHLC files")
        else:
            logging.error("No OHLC files to combine!")
    
    logging.info("✅ All OHLC processing completed!")

if __name__ == '__main__':
    main()