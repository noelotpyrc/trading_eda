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
    # Configuration for uploading enriched coin metadata
    db_file = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    csv_file = 'post2024_meme_coins_enriched.csv'  # Path to enriched metadata CSV
    table_name = 'coin_meta'

    print("🚀 Uploading enriched coin metadata to DuckDB...")
    print("=" * 60)
    print(f"Database: {db_file}")
    print(f"CSV file: {csv_file}")
    print(f"Table: {table_name}")
    print()

    try:
        # Connect to the database
        connection = connect_to_duckdb(db_file)
        print("✅ Connected to DuckDB")

        # Check if CSV file exists
        if not os.path.exists(csv_file):
            print(f"❌ Error: {csv_file} not found")
            print("💡 Run the DexPaprika enrichment script first to generate the enriched CSV")
            connection.close()
            exit(1)

        # Upload the enriched CSV
        upload_csv_to_duckdb(connection, csv_file, table_name)
        print("✅ Successfully uploaded enriched metadata")

        # Verify the upload and show summary
        result = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        print(f"📊 Total rows in {table_name}: {result[0]:,}")
        
        # Show successful enrichments
        successful_count = connection.execute(f"SELECT COUNT(*) FROM {table_name} WHERE api_status = 'success'").fetchone()
        print(f"🎯 Successfully enriched tokens: {successful_count[0]:,}")
        
        # Show sample of enriched data
        print(f"\n🏆 SAMPLE ENRICHED TOKENS:")
        sample_query = f"""
        SELECT symbol, name, price_usd, fdv, liquidity_usd 
        FROM {table_name} 
        WHERE api_status = 'success' 
        ORDER BY fdv DESC 
        LIMIT 10
        """
        sample_results = connection.execute(sample_query).fetchall()
        
        for i, (symbol, name, price_usd, fdv, liquidity_usd) in enumerate(sample_results, 1):
            print(f"{i:2d}. {symbol:<8} ({name[:30]:<30}) | "
                  f"${price_usd:>8.6f} | "
                  f"FDV: ${fdv:>12,.0f} | "
                  f"Liquidity: ${liquidity_usd:>12,.0f}")

        # Close the connection
        connection.close()
        print(f"\n✅ Upload completed successfully!")
        print(f"💡 Table '{table_name}' is now available for analysis in DuckDB")
        
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        if 'connection' in locals():
            connection.close()
