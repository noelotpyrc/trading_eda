import duckdb
import pandas as pd
import os
import glob

def connect_to_duckdb(db_file):
    """
    Connect to a DuckDB database.
    """
    con = duckdb.connect(database=db_file, read_only=False)
    return con

def upload_csv_to_duckdb(con, csv_file, table_name):
    """
    Upload a CSV file to a DuckDB table.
    """
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")
    print(f"Successfully uploaded {csv_file} to table {table_name}")

def upload_csvs_from_folder_to_duckdb(con, folder_path, table_name):
    """
    Upload and append all CSV files from a folder to a DuckDB table.
    """
    full_folder_path = os.path.abspath(folder_path)
    glob_pattern = os.path.join(full_folder_path, '*.csv')
    
    # Get all matching file paths
    all_files = glob.glob(glob_pattern)
    
    # Filter out hidden macOS files that start with '._'
    csv_files = [f for f in all_files if not os.path.basename(f).startswith('._')]

    if not csv_files:
        print(f"No valid CSV files found in {folder_path}")
        return
        
    print(f"Creating table '{table_name}' and ingesting {len(csv_files)} CSVs from {folder_path}...")
    # DuckDB's read_csv_auto can take a list of files. Using parameter binding for safety.
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?)", [csv_files])
    print(f"Successfully created table and ingested all CSVs from {folder_path}")

if __name__ == '__main__':
    # Configuration for uploading deduplicated first day trades
    db_file = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    folder_path = '/Volumes/Extreme SSD/trading_data/solana/first_day_dedup'  # Path to deduplicated CSV files
    table_name = 'first_day_trades'

    print("🚀 Uploading deduplicated first day trades to DuckDB...")
    print("=" * 60)
    print(f"Database: {db_file}")
    print(f"Folder: {folder_path}")
    print(f"Table: {table_name}")
    print()

    try:
        # Connect to the database
        connection = connect_to_duckdb(db_file)
        print("✅ Connected to DuckDB")

        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"❌ Error: {folder_path} not found")
            print("💡 Run the CSV deduplicator script first to generate deduplicated files")
            connection.close()
            exit(1)

        # Check if there are CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        csv_files = [f for f in csv_files if not os.path.basename(f).startswith('._')]  # Filter out hidden files
        
        if not csv_files:
            print(f"❌ Error: No CSV files found in {folder_path}")
            print("💡 Run the CSV deduplicator script first to generate deduplicated files")
            connection.close()
            exit(1)
            
        print(f"📁 Found {len(csv_files)} deduplicated CSV files")

        # Upload all deduplicated CSVs to create the table
        upload_csvs_from_folder_to_duckdb(connection, folder_path, table_name)
        print("✅ Successfully uploaded deduplicated trades")

        # Verify the upload and show summary
        result = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        print(f"📊 Total rows in {table_name}: {result[0]:,}")
        
        # Show unique traders count
        unique_traders = connection.execute(f"SELECT COUNT(DISTINCT swapper) FROM {table_name}").fetchone()
        print(f"👥 Unique traders: {unique_traders[0]:,}")
        
        # Show unique coins count
        unique_coins = connection.execute(f"SELECT COUNT(DISTINCT mint) FROM {table_name}").fetchone()
        print(f"🪙 Unique coins: {unique_coins[0]:,}")
        
        # Show date range
        date_range = connection.execute(f"""
        SELECT 
            MIN(DATE(block_timestamp)) as earliest_date,
            MAX(DATE(block_timestamp)) as latest_date
        FROM {table_name}
        """).fetchone()
        print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
        
        # Show sample of data
        print(f"\n🏆 SAMPLE DEDUPLICATED TRADES:")
        sample_query = f"""
        SELECT swapper, mint, block_timestamp, swap_from_amount, swap_to_amount
        FROM {table_name} 
        ORDER BY block_timestamp DESC 
        LIMIT 5
        """
        sample_results = connection.execute(sample_query).fetchall()
        
        for i, (swapper, mint, timestamp, from_amt, to_amt) in enumerate(sample_results, 1):
            print(f"{i}. {swapper[:20]}... | {mint[:20]}... | {timestamp} | {from_amt:,.0f} → {to_amt:,.0f}")

        # Close the connection
        connection.close()
        print(f"\n✅ Upload completed successfully!")
        print(f"💡 Table '{table_name}' is now available for analysis in DuckDB")
        print(f"🎯 Ready for trader profiling and diversification analysis!")
        
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        if 'connection' in locals():
            connection.close()
