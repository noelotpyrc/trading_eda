import duckdb
import pandas as pd
import numpy as np
from data.duckdb_data_manager import connect_to_duckdb

def inspect_database(db_file):
    """
    Comprehensive inspection of the DuckDB database
    """
    print("=" * 60)
    print("DUCKDB DATABASE INSPECTION")
    print("=" * 60)
    
    # Connect to database
    con = connect_to_duckdb(db_file)
    
    try:
        # 1. List all tables
        print("\n1. AVAILABLE TABLES:")
        print("-" * 30)
        tables = con.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        print(f"Found {len(table_names)} tables: {table_names}")
        
        # 2. For each table, get detailed information
        for table_name in table_names:
            print(f"\n2. TABLE: {table_name}")
            print("-" * 30)
            
            # Table schema
            print("\nSchema:")
            schema = con.execute(f"DESCRIBE {table_name}").fetchall()
            schema_df = pd.DataFrame(schema, columns=['Column', 'Type', 'Null', 'Key', 'Default', 'Extra'])
            print(schema_df.to_string(index=False))
            
            # Row count
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"\nTotal rows: {row_count:,}")
            
            # Skip unique row analysis for very large datasets
            if row_count > 100000000:
                print("Unique row analysis skipped (dataset too large)")
            
            # Sample data
            print(f"\nFirst 5 rows:")
            sample = con.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
            print(sample.to_string(index=False))
            
            # Date range if there are date/timestamp columns
            date_columns = [col for col, dtype in zip(schema_df['Column'], schema_df['Type']) 
                           if 'timestamp' in dtype.lower() or 'date' in dtype.lower()]
            
            if date_columns:
                print(f"\nDate/Time Range Analysis:")
                for date_col in date_columns:
                    date_range = con.execute(f"""
                        SELECT 
                            MIN({date_col}) as earliest,
                            MAX({date_col}) as latest,
                            COUNT(DISTINCT DATE({date_col})) as unique_days
                        FROM {table_name}
                        WHERE {date_col} IS NOT NULL
                    """).fetchone()
                    print(f"{date_col}: {date_range[0]} to {date_range[1]} ({date_range[2]} unique days)")
            
            # Basic value counts for categorical columns (limited sample)
            categorical_cols = ['mint', 'swapper', 'swap_from_mint', 'swap_to_mint', 'succeeded']
            for col in categorical_cols:
                if col in [row[0] for row in schema]:
                    print(f"\nTop 10 values for {col}:")
                    try:
                        top_values = con.execute(f"""
                            SELECT {col}, COUNT(*) as count 
                            FROM {table_name} 
                            WHERE {col} IS NOT NULL
                            GROUP BY {col} 
                            ORDER BY count DESC 
                            LIMIT 10
                        """).fetchdf()
                        print(top_values.to_string(index=False))
                    except Exception as e:
                        print(f"Could not analyze {col}: {e}")
            
            print("\n" + "="*60)
        
        return table_names, con
        
    except Exception as e:
        print(f"Error during inspection: {e}")
        con.close()
        raise

def get_data_sample(con, table_name, sample_size=1000):
    """
    Get a random sample of data for initial exploration
    """
    sample_df = con.execute(f"""
        SELECT * FROM {table_name} 
        USING SAMPLE {sample_size} ROWS
    """).fetchdf()
    return sample_df

if __name__ == '__main__':
    # Update this path to match your actual database location
    db_file = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    
    # Inspect the database
    table_names, connection = inspect_database(db_file)
    
    # Get a sample for further analysis
    if table_names:
        main_table = table_names[0]  # Assume first table is main one
        print(f"\nGetting sample data from {main_table}...")
        sample_data = get_data_sample(connection, main_table)
        print(f"Sample shape: {sample_data.shape}")
        
        # Save sample for notebooks
        sample_data.to_csv('data/sample_trades.csv', index=False)
        print("Sample data saved to 'data/sample_trades.csv'")
    
    connection.close() 