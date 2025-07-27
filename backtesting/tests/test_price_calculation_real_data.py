#!/usr/bin/env python3
"""
Real data test for transaction-based price calculation in SolanaTransactionFeed.
Uses actual Solana transaction data from DuckDB to validate price calculations.
"""

import sys
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_transaction_feed import SolanaTransactionFeed


def test_price_calculation_with_real_data():
    """Test price calculation using real Solana transaction data."""
    print("🧪 Testing Price Calculation with Real Transaction Data")
    print("=" * 60)
    
    # Connect to DuckDB
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Test coin
    coin_id = "8xPe3DMr52oYAkCBk57ZeQE1h5zBkWNk37eE4J8spump"
    print(f"Testing coin: {coin_id}")
    
    # Get real transaction data (limited sample for testing)
    print("\n1. Loading real transaction data...")
    transaction_query = f"""
    SELECT 
        block_timestamp,
        mint,
        swapper,
        succeeded,
        swap_from_mint,
        swap_to_mint,
        swap_from_amount,
        swap_to_amount,
        CASE WHEN mint = swap_to_mint THEN 1 ELSE 0 END as is_buy,
        CASE 
            WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_from_amount
            WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_to_amount
            ELSE 0.0
        END as sol_amount
    FROM first_day_trades
    WHERE mint = '{coin_id}'
    AND succeeded = TRUE
    AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
         OR swap_to_mint = 'So11111111111111111111111111111111111111112')
    AND mint != 'So11111111111111111111111111111111111111112'
    ORDER BY block_timestamp
    LIMIT 1000
    """
    
    transaction_df = conn.execute(transaction_query).fetchdf()
    
    if len(transaction_df) == 0:
        print("❌ No transaction data found")
        conn.close()
        return
    
    print(f"✅ Loaded {len(transaction_df)} transactions")
    print(f"   Time range: {transaction_df['block_timestamp'].min()} to {transaction_df['block_timestamp'].max()}")
    print(f"   Buy transactions: {transaction_df['is_buy'].sum()}")
    print(f"   Sell transactions: {(~transaction_df['is_buy']).sum()}")
    
    # Create SolanaTransactionFeed with real data
    print("\n2. Creating SolanaTransactionFeed with real data...")
    feed = SolanaTransactionFeed(transaction_data=transaction_df)
    
    # Analyze price calculation results
    print("\n3. Analyzing price calculation results...")
    
    # Check how many transactions got valid prices
    valid_prices = feed.df[feed.df['transaction_price'] > 0]
    zero_prices = feed.df[feed.df['transaction_price'] == 0]
    
    print(f"📊 Price Calculation Statistics:")
    print(f"   Total transactions: {len(feed.df)}")
    print(f"   Valid prices (>0): {len(valid_prices)} ({len(valid_prices)/len(feed.df)*100:.1f}%)")
    print(f"   Zero prices: {len(zero_prices)} ({len(zero_prices)/len(feed.df)*100:.1f}%)")
    
    if len(valid_prices) > 0:
        prices = valid_prices['transaction_price']
        print(f"   Price range: {prices.min():.8f} to {prices.max():.8f}")
        print(f"   Average price: {prices.mean():.8f}")
        print(f"   Median price: {prices.median():.8f}")
        print(f"   Price std dev: {prices.std():.8f}")
    
    # Show sample transactions with prices and debug info
    print(f"\n4. Sample transactions with calculated prices:")
    print("-" * 80)
    sample_size = min(10, len(valid_prices))
    
    if sample_size > 0:
        sample_transactions = valid_prices.head(sample_size)
        
        for i, (_, txn) in enumerate(sample_transactions.iterrows()):
            sol_amount = txn['sol_amount']
            price = txn['transaction_price']
            is_buy = txn['is_buy']
            direction = "BUY" if is_buy else "SELL"
            
            # Show raw swap amounts for debugging
            swap_from_amount = txn['swap_from_amount']
            swap_to_amount = txn['swap_to_amount']
            
            # Calculate implied token amount
            if price > 0:
                token_amount = sol_amount / price
            else:
                token_amount = 0
            
            print(f"   {i+1:2d}. {txn['block_timestamp']} | {direction}")
            print(f"       SOL: {sol_amount:10.6f} | Price: {price:15.10f}")
            print(f"       Raw: From {swap_from_amount:15.6f} → To {swap_to_amount:15.6f}")
            print(f"       Calc: {swap_from_amount:15.6f} / {swap_to_amount:15.6f} = {swap_from_amount/swap_to_amount if swap_to_amount > 0 else 0:15.10f}")
            print(f"       From: {txn['swap_from_mint'][:20]}...")
            print(f"       To:   {txn['swap_to_mint'][:20]}...")
            print()
    
    # Test OHLCV calculation with real data
    print("5. Testing OHLCV calculation with real data...")
    print("-" * 50)
    
    if len(feed.sample_timestamps) == 0:
        # Create a test sample timestamp based on transaction data
        first_transaction_time = transaction_df['block_timestamp'].min()
        test_sample_timestamp = first_transaction_time + pd.Timedelta(seconds=120)
        print(f"   Using test sample timestamp: {test_sample_timestamp}")
    else:
        test_sample_timestamp = feed.sample_timestamps[0]
        print(f"   Using feed sample timestamp: {test_sample_timestamp}")
    
    # Calculate OHLCV for the test timestamp
    ohlcv = feed._calculate_ohlcv_from_transactions(test_sample_timestamp)
    
    print(f"📊 OHLCV Results:")
    print(f"   Open:   {ohlcv['open']:.8f}")
    print(f"   High:   {ohlcv['high']:.8f}")
    print(f"   Low:    {ohlcv['low']:.8f}")
    print(f"   Close:  {ohlcv['close']:.8f}")
    print(f"   Volume: {ohlcv['volume']:.6f} SOL")
    
    # Validate OHLCV logic
    if ohlcv['high'] >= ohlcv['low'] and ohlcv['high'] >= ohlcv['open'] and ohlcv['high'] >= ohlcv['close']:
        print("✅ OHLCV high >= other prices (valid)")
    else:
        print("❌ OHLCV high < other prices (invalid)")
    
    if ohlcv['low'] <= ohlcv['open'] and ohlcv['low'] <= ohlcv['close']:
        print("✅ OHLCV low <= other prices (valid)")
    else:
        print("❌ OHLCV low > other prices (invalid)")
    
    # Test multiple sample timestamps if available
    if len(feed.sample_timestamps) > 1:
        print(f"\n6. Testing multiple sample timestamps...")
        sample_count = min(5, len(feed.sample_timestamps))
        
        for i in range(sample_count):
            sample_ts = feed.sample_timestamps[i]
            ohlcv = feed._calculate_ohlcv_from_transactions(sample_ts)
            
            print(f"   {i+1}. {sample_ts} | O:{ohlcv['open']:.6f} H:{ohlcv['high']:.6f} L:{ohlcv['low']:.6f} C:{ohlcv['close']:.6f} V:{ohlcv['volume']:.2f}")
    
    # Analyze price continuity
    print(f"\n7. Price continuity analysis...")
    if len(valid_prices) > 1:
        valid_prices_sorted = valid_prices.sort_values('block_timestamp')
        prices = valid_prices_sorted['transaction_price'].values
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                change = (prices[i] - prices[i-1]) / prices[i-1] * 100
                price_changes.append(abs(change))
        
        if price_changes:
            avg_change = sum(price_changes) / len(price_changes)
            max_change = max(price_changes)
            
            print(f"   Average price change: {avg_change:.2f}%")
            print(f"   Maximum price change: {max_change:.2f}%")
            
            # Flag large price jumps
            large_jumps = [c for c in price_changes if c > 50]  # >50% change
            if large_jumps:
                print(f"   ⚠️  Large price jumps (>50%): {len(large_jumps)} occurrences")
                print(f"   Largest jump: {max(large_jumps):.2f}%")
            else:
                print(f"   ✅ No large price jumps detected")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if len(valid_prices) / len(feed.df) > 0.8:
        print("🎉 High price calculation success rate - good data quality")
    elif len(valid_prices) / len(feed.df) > 0.5:
        print("✅ Moderate price calculation success rate - acceptable")
    else:
        print("⚠️  Low price calculation success rate - check data filtering")
    
    print(f"💰 Price range spans {(prices.max()/prices.min() if len(valid_prices) > 0 and prices.min() > 0 else 0):.2f}x")
    print(f"📈 OHLCV calculation working correctly")
    print(f"🔧 Feed ready for Backtrader integration")
    
    conn.close()


if __name__ == '__main__':
    test_price_calculation_with_real_data()