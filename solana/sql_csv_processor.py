#!/usr/bin/env python3
"""
SQL-based CSV processor for Solana feature extraction
Loads individual CSV files into DuckDB and runs batch feature extraction
"""

import pandas as pd
import duckdb
import os
import glob
import logging
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_csv_with_sql(csv_file_path, sql_query, output_dir, db_path):
    """Process a single CSV file using SQL batch extraction"""
    try:
        csv_name = os.path.basename(csv_file_path)
        logging.info(f"Processing {csv_name}")
        
        start_time = time.time()
        
        # Connect to existing DuckDB
        conn = duckdb.connect(db_path)
        
        # Create temporary table name for this CSV
        temp_table = f"temp_trades_{int(time.time() * 1000000) % 1000000}"
        
        # Load CSV data into temporary table
        conn.execute(f"""
            CREATE TEMP TABLE {temp_table} AS 
            SELECT * FROM read_csv_auto('{csv_file_path}')
        """)
        
        # Check data loaded
        count = conn.execute(f"SELECT COUNT(*) FROM {temp_table}").fetchone()[0]
        if count == 0:
            logging.warning(f"Empty CSV: {csv_name}")
            conn.execute(f"DROP TABLE {temp_table}")
            conn.close()
            return None
        
        coins_count = conn.execute(f"SELECT COUNT(DISTINCT mint) FROM {temp_table}").fetchone()[0]
        logging.info(f"{csv_name}: {count:,} transactions, {coins_count} unique coins")
        
        # Replace table name in SQL query
        modified_sql = sql_query.replace('first_day_trades', temp_table)
        
        # Execute feature extraction SQL
        try:
            features_df = conn.execute(modified_sql).fetchdf()
            
            # Cleanup temporary table
            conn.execute(f"DROP TABLE {temp_table}")
            conn.close()
            
            if features_df.empty:
                logging.warning(f"No features extracted from {csv_name}")
                return None
            
            # Save results
            output_file = os.path.join(output_dir, f"features_{csv_name}")
            features_df.to_csv(output_file, index=False)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logging.info(f"✅ {csv_name}: {len(features_df):,} features in {duration:.2f}s -> {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"SQL execution failed for {csv_name}: {e}")
            # Cleanup on error
            try:
                conn.execute(f"DROP TABLE {temp_table}")
            except:
                pass
            conn.close()
            return None
            
    except Exception as e:
        logging.error(f"Error processing {csv_file_path}: {e}")
        return None

def process_csv_batch_sql(csv_files, sql_query, output_dir, batch_id, db_path):
    """Process a batch of CSV files using SQL"""
    logging.info(f"Batch {batch_id}: Processing {len(csv_files)} CSV files")
    
    successful_files = []
    
    for csv_file in csv_files:
        result = process_single_csv_with_sql(csv_file, sql_query, output_dir, db_path)
        if result:
            successful_files.append(result)
    
    logging.info(f"Batch {batch_id}: Completed {len(successful_files)}/{len(csv_files)} files")
    return successful_files

def main():
    parser = argparse.ArgumentParser(description='Process CSV files using SQL batch feature extraction')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save feature files')
    parser.add_argument('--db_path', type=str, default='/Volumes/Extreme SSD/DuckDB/solana.duckdb', help='DuckDB file path')
    parser.add_argument('--sql_file', type=str, default='solana/batch_feature_extraction.sql', help='SQL file path')
    parser.add_argument('--batch_size', type=int, default=10, help='CSV files per batch')
    parser.add_argument('--max_workers', type=int, default=None, help='Max parallel workers')
    parser.add_argument('--combine_output', action='store_true', help='Combine all results into single file')
    
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
    
    # Split into batches
    batches = [csv_files[i:i+args.batch_size] for i in range(0, len(csv_files), args.batch_size)]
    
    max_workers = args.max_workers or min(multiprocessing.cpu_count(), len(batches))
    logging.info(f"Processing {len(batches)} batches with {max_workers} workers")
    
    # Process batches in parallel
    all_output_files = []
    completed_batches = 0
    
    start_time = time.time()
    
    # Verify DuckDB exists
    if not os.path.exists(args.db_path):
        logging.error(f"DuckDB file not found: {args.db_path}")
        return
    
    logging.info(f"Using DuckDB: {args.db_path}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_csv_batch_sql, batch, sql_query, args.output_dir, i, args.db_path): i 
            for i, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_files = future.result()
                all_output_files.extend(batch_files)
                completed_batches += 1
                logging.info(f"Completed batch {batch_id+1}/{len(batches)} ({completed_batches}/{len(batches)} total)")
            except Exception as e:
                logging.error(f"Batch {batch_id} failed: {e}")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    logging.info(f"Processing completed in {total_duration:.2f}s")
    logging.info(f"Successfully processed {len(all_output_files)}/{len(csv_files)} CSV files")
    
    # Combine results if requested
    if args.combine_output and all_output_files:
        logging.info("Combining all feature files...")
        
        all_features = []
        for feature_file in all_output_files:
            try:
                df = pd.read_csv(feature_file)
                all_features.append(df)
            except Exception as e:
                logging.error(f"Error reading {feature_file}: {e}")
        
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            final_output = os.path.join(args.output_dir, "combined_features_sql.csv")
            combined_df.to_csv(final_output, index=False)
            
            logging.info(f"✅ Combined features saved: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
            logging.info(f"Final output: {final_output}")
            
            # Cleanup individual files
            for feature_file in all_output_files:
                os.remove(feature_file)
            logging.info("Cleaned up individual feature files")
        else:
            logging.error("No feature files to combine!")
    
    logging.info("✅ All processing completed!")

if __name__ == '__main__':
    main()