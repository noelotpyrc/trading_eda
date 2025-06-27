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
    # Example usage
    db_file = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    folder_path = '/Volumes/Extreme SSD/trading_data/solana/first_day'
    table_name = 'first_day_trades'

    # Connect to the database
    connection = connect_to_duckdb(db_file)

    # Upload all CSVs from the folder
    upload_csvs_from_folder_to_duckdb(connection, folder_path, table_name)

    # Verify the upload
    result = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    print(f"Verification: Total rows in {table_name} is {result[0]}")

    # Close the connection
    connection.close()
