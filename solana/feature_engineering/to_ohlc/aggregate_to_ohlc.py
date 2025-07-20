#!/usr/bin/env python3
"""
Aggregate Solana first_day_trades data to OHLC format similar to NVDA data.
Each row represents one coin at one block timestamp level.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
SOL_MINT = 'So11111111111111111111111111111111111111112'
INPUT_FILE = "data/solana/first_day_trades/first_day_trades_batch_578.csv"
OUTPUT_DIR = Path("data/solana/ohlc_aggregated")

def load_solana_data(file_path):
    """Load and prepare Solana trading data"""
    print(f"Loading Solana data from {file_path}...")
    
    df = pd.read_csv(file_path)
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'])
    
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Date range: {df['block_timestamp'].min()} to {df['block_timestamp'].max()}")
    print(f"   Unique coins: {df['mint'].nunique()}")
    
    return df

def prepare_trading_data(df):
    """Prepare trading data with price and volume calculations"""
    print("Preparing trading data...")
    
    # Filter to successful SOL-related trades only
    df_clean = df[
        (df['succeeded'] == True) &
        ((df['swap_from_mint'] == SOL_MINT) | (df['swap_to_mint'] == SOL_MINT)) &
        (df['mint'] != SOL_MINT)
    ].copy()
    
    print(f"✅ Filtered to {len(df_clean):,} successful SOL trades")
    
    # Add trading direction indicators
    df_clean['is_buy'] = df_clean['mint'] == df_clean['swap_to_mint']
    df_clean['is_sell'] = df_clean['mint'] == df_clean['swap_from_mint']
    
    # Calculate SOL amounts (denominator for price calculation)
    df_clean['sol_amount'] = 0.0
    buy_mask = df_clean['is_buy'] & (df_clean['swap_from_mint'] == SOL_MINT)
    sell_mask = df_clean['is_sell'] & (df_clean['swap_to_mint'] == SOL_MINT)
    
    df_clean.loc[buy_mask, 'sol_amount'] = df_clean.loc[buy_mask, 'swap_from_amount']
    df_clean.loc[sell_mask, 'sol_amount'] = df_clean.loc[sell_mask, 'swap_to_amount']
    
    # Calculate token amounts (numerator for price calculation)
    df_clean['token_amount'] = 0.0
    df_clean.loc[buy_mask, 'token_amount'] = df_clean.loc[buy_mask, 'swap_to_amount']
    df_clean.loc[sell_mask, 'token_amount'] = df_clean.loc[sell_mask, 'swap_from_amount']
    
    # Calculate price as SOL per token
    # Price = SOL_amount / token_amount
    df_clean['price'] = df_clean['sol_amount'] / (df_clean['token_amount'] + 1e-18)
    
    # Filter out invalid prices (too high/low or zero)
    df_clean = df_clean[
        (df_clean['sol_amount'] > 0) &
        (df_clean['token_amount'] > 0)
    ].copy()
    
    print(f"✅ After price filtering: {len(df_clean):,} valid trades")
    print(f"   Price range: {df_clean['price'].min():.2e} to {df_clean['price'].max():.2e} SOL per token")
    
    return df_clean

def aggregate_to_ohlc(df_clean):
    """Aggregate transaction data to OHLC format by coin and block timestamp"""
    print("Aggregating to OHLC format...")
    
    # Group by coin and block timestamp
    agg_data = []
    
    # Process each coin separately for better memory management
    unique_coins = df_clean['mint'].unique()
    print(f"Processing {len(unique_coins)} coins...")
    
    for i, coin in enumerate(unique_coins, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(unique_coins)} coins...")
        
        coin_data = df_clean[df_clean['mint'] == coin].copy()
        
        # Group by block timestamp
        grouped = coin_data.groupby('block_timestamp').agg({
            'price': ['first', 'max', 'min', 'last', 'count'],  # OHLC + count
            'sol_amount': ['sum'],  # Total SOL volume
            'token_amount': ['sum'],  # Total token volume
            '__row_index': ['count']  # Transaction count (using any column for count)
        }).round(12)
        
        # Flatten column names
        grouped.columns = ['open', 'high', 'low', 'close', 'price_count', 'sol_volume', 'token_volume', 'transactions']
        
        # Calculate VWAP (Volume Weighted Average Price)
        # VWAP = sum(price * sol_volume) / sum(sol_volume) for each timestamp
        coin_data_vwap = coin_data.groupby('block_timestamp').apply(
            lambda x: (x['price'] * x['sol_amount']).sum() / x['sol_amount'].sum()
        ).rename('vwap')
        
        # Merge VWAP with OHLC data
        grouped = grouped.join(coin_data_vwap)
        
        # Add metadata
        grouped['mint'] = coin
        grouped['timestamp_unix'] = (grouped.index.astype('int64') // 1e6).astype('int64')  # Convert to milliseconds
        grouped['time_readable'] = grouped.index.strftime('%Y%m%d %H:%M:%S')
        
        # Reset index to make block_timestamp a column
        grouped = grouped.reset_index()
        
        agg_data.append(grouped)
    
    # Combine all coins
    ohlc_df = pd.concat(agg_data, ignore_index=True)
    
    print(f"✅ Created OHLC data: {len(ohlc_df):,} records")
    print(f"   Time range: {ohlc_df['block_timestamp'].min()} to {ohlc_df['block_timestamp'].max()}")
    print(f"   Average transactions per record: {ohlc_df['transactions'].mean():.1f}")
    
    return ohlc_df

def format_output_columns(ohlc_df):
    """Format columns to match NVDA structure"""
    print("Formatting output columns...")
    
    # Reorder and rename columns to match NVDA format
    output_df = ohlc_df[[
        'time_readable',
        'timestamp_unix', 
        'open',
        'high',
        'low',
        'close',
        'vwap',
        'sol_volume',
        'token_volume',
        'transactions',
        'mint',
        'block_timestamp'
    ]].copy()
    
    # Rename columns to match NVDA format (with additions for crypto)
    output_df.columns = [
        'time',
        'timestamp',
        'open',
        'high', 
        'low',
        'close',
        'vwap',
        'sol_volume',
        'token_volume',
        'transactions',
        'mint',
        'block_timestamp_iso'
    ]
    
    # Add empty OTC column for compatibility
    output_df['otc'] = ''
    
    # Round numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'sol_volume', 'token_volume']
    output_df[numeric_cols] = output_df[numeric_cols].round(12)
    
    print(f"✅ Formatted output: {len(output_df)} records with {len(output_df.columns)} columns")
    
    return output_df

def save_output_files(output_df, output_dir):
    """Save aggregated data to files"""
    print(f"Saving output files to {output_dir}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined file
    combined_file = output_dir / "all_coins_ohlc.csv"
    output_df.to_csv(combined_file, index=False)
    print(f"✅ Saved combined file: {combined_file}")
    
    # Save individual files per coin (similar to NVDA structure)
    coin_files_created = 0
    unique_coins = output_df['mint'].unique()
    
    for coin in unique_coins:
        coin_data = output_df[output_df['mint'] == coin]
        
        # Create filename with first 8 characters of mint
        coin_short = coin[:8]
        coin_file = output_dir / f"{coin_short}_ohlc.csv"
        
        # Save coin-specific data
        coin_data.to_csv(coin_file, index=False)
        coin_files_created += 1
        
        if coin_files_created <= 5:  # Show first few files
            print(f"  Created: {coin_file} ({len(coin_data)} records)")
    
    print(f"✅ Created {coin_files_created} individual coin files")
    
    # Save summary statistics
    summary_file = output_dir / "summary_stats.txt"
    with open(summary_file, 'w') as f:
        f.write("Solana OHLC Aggregation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total records: {len(output_df):,}\n")
        f.write(f"Unique coins: {output_df['mint'].nunique()}\n")
        f.write(f"Time range: {output_df['block_timestamp_iso'].min()} to {output_df['block_timestamp_iso'].max()}\n")
        f.write(f"Average price range per coin:\n")
        f.write(f"  Min price: {output_df['low'].min():.2e} SOL\n")
        f.write(f"  Max price: {output_df['high'].max():.2e} SOL\n")
        f.write(f"Total SOL volume: {output_df['sol_volume'].sum():,.2f}\n")
        f.write(f"Total transactions: {output_df['transactions'].sum():,}\n")
        f.write(f"\nTop 10 coins by SOL volume:\n")
        
        top_coins = output_df.groupby('mint')['sol_volume'].sum().sort_values(ascending=False).head(10)
        for mint, volume in top_coins.items():
            f.write(f"  {mint[:12]}...: {volume:,.2f} SOL\n")
    
    print(f"✅ Saved summary: {summary_file}")

def show_sample_output(output_df):
    """Display sample output for verification"""
    print("\n" + "="*80)
    print("SAMPLE OUTPUT (first 5 records):")
    print("="*80)
    
    sample = output_df.head()
    for col in output_df.columns:
        print(f"{col}: {sample[col].iloc[0]}")
        if col == 'close':  # Stop after showing key price data
            break
    
    print("\nColumn Summary:")
    print(f"Total columns: {len(output_df.columns)}")
    print(f"Columns: {list(output_df.columns)}")
    
    print(f"\nData types:")
    print(output_df.dtypes)

def main():
    """Main aggregation function"""
    print("="*60)
    print("SOLANA TO OHLC AGGREGATION")
    print("="*60)
    
    # Load data
    try:
        df = load_solana_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("Please adjust the INPUT_FILE path in the script")
        return
    
    # Prepare trading data
    df_clean = prepare_trading_data(df)
    
    if len(df_clean) == 0:
        print("❌ No valid trading data found after filtering")
        return
    
    # Aggregate to OHLC
    ohlc_df = aggregate_to_ohlc(df_clean)
    
    # Format output
    output_df = format_output_columns(ohlc_df)
    
    # Save files
    save_output_files(output_df, OUTPUT_DIR)
    
    # Show sample
    show_sample_output(output_df)
    
    print("\n" + "="*60)
    print("✅ AGGREGATION COMPLETE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Combined file: {OUTPUT_DIR}/all_coins_ohlc.csv")
    print("="*60)

if __name__ == "__main__":
    main()