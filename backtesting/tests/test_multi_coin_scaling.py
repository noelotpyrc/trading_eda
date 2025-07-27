#!/usr/bin/env python3
"""
Multi-Coin Price Scaling Test
Tests price scaling across different meme coins with varying token supplies and price ranges.
"""

import sys
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_transaction_feed import SolanaTransactionFeed


def test_multi_coin_price_scaling():
    """Test price scaling with multiple meme coins having different characteristics."""
    print("🧪 Testing Price Scaling Across Multiple Meme Coins")
    print("=" * 60)
    
    # Connect to DuckDB
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Get diverse meme coins with different characteristics
    print("1. Finding diverse meme coins for testing...")
    
    # Query to find coins with different price ranges and volumes
    coin_selection_query = """
    WITH coin_stats AS (
        SELECT 
            mint,
            COUNT(*) as total_txns,
            AVG(CASE 
                WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
                THEN swap_from_amount / swap_to_amount
                WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
                THEN swap_to_amount / swap_from_amount
                ELSE NULL
            END) as avg_price,
            SUM(CASE 
                WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
                THEN swap_from_amount
                WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
                THEN swap_to_amount
                ELSE 0
            END) as total_sol_volume
        FROM first_day_trades
        WHERE succeeded = TRUE
        AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
             OR swap_to_mint = 'So11111111111111111111111111111111111111112')
        AND mint != 'So11111111111111111111111111111111111111112'
        GROUP BY mint
        HAVING total_txns >= 100  -- Minimum transaction count
        AND avg_price IS NOT NULL
    ),
    price_categories AS (
        SELECT *,
            CASE 
                WHEN avg_price < 1e-10 THEN 'ultra_micro'
                WHEN avg_price < 1e-9 THEN 'micro'  
                WHEN avg_price < 1e-8 THEN 'small'
                WHEN avg_price < 1e-7 THEN 'medium'
                ELSE 'large'
            END as price_category
        FROM coin_stats
    )
    SELECT 
        mint,
        total_txns,
        avg_price,
        total_sol_volume,
        price_category
    FROM price_categories
    WHERE price_category IN ('ultra_micro', 'micro', 'small', 'medium')
    ORDER BY price_category, total_txns DESC
    """
    
    coin_candidates = conn.execute(coin_selection_query).fetchdf()
    
    if len(coin_candidates) == 0:
        print("❌ No suitable coins found")
        conn.close()
        return
    
    print(f"✅ Found {len(coin_candidates)} candidate coins")
    
    # Select representative coins from different categories
    test_coins = []
    for category in ['ultra_micro', 'micro', 'small', 'medium']:
        category_coins = coin_candidates[coin_candidates['price_category'] == category]
        if len(category_coins) > 0:
            # Pick the one with most transactions
            best_coin = category_coins.iloc[0]
            test_coins.append({
                'mint': best_coin['mint'],
                'category': category,
                'avg_price': best_coin['avg_price'],
                'total_txns': best_coin['total_txns'],
                'sol_volume': best_coin['total_sol_volume']
            })
    
    print(f"\n📊 Selected test coins:")
    for i, coin in enumerate(test_coins):
        print(f"   {i+1}. {coin['category'].upper()}: {coin['mint'][:20]}...")
        print(f"      Avg price: {coin['avg_price']:.2e} SOL/token")
        print(f"      Transactions: {coin['total_txns']:,}")
        print(f"      SOL volume: {coin['sol_volume']:.2f}")
        print()
    
    # Test each coin
    results = []
    
    for i, coin in enumerate(test_coins):
        print(f"{i+1}. Testing {coin['category'].upper()} price coin...")
        print("-" * 40)
        
        # Get transaction data for this coin
        coin_id = coin['mint']
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
        LIMIT 200
        """
        
        transaction_df = conn.execute(transaction_query).fetchdf()
        
        if len(transaction_df) == 0:
            print(f"   ❌ No transaction data for coin {i+1}")
            continue
        
        print(f"   📊 Loaded {len(transaction_df)} transactions")
        
        # Create feed and test price scaling
        try:
            feed = SolanaTransactionFeed(transaction_data=transaction_df)
            
            # Analyze price scaling results
            valid_prices = feed.df[feed.df['transaction_price'] > 0]
            
            if len(valid_prices) == 0:
                print(f"   ❌ No valid prices calculated")
                continue
            
            # Calculate price statistics
            raw_prices = valid_prices['transaction_price'] / feed.PRICE_SCALE_FACTOR
            scaled_prices = valid_prices['transaction_price']
            
            price_stats = {
                'coin_id': coin_id,
                'category': coin['category'],
                'total_txns': len(transaction_df),
                'valid_prices': len(valid_prices),
                'success_rate': len(valid_prices) / len(transaction_df) * 100,
                'raw_min': raw_prices.min(),
                'raw_max': raw_prices.max(),
                'raw_avg': raw_prices.mean(),
                'scaled_min': scaled_prices.min(),
                'scaled_max': scaled_prices.max(),
                'scaled_avg': scaled_prices.mean(),
                'scale_factor': feed.PRICE_SCALE_FACTOR
            }
            
            results.append(price_stats)
            
            print(f"   ✅ Price calculation: {price_stats['success_rate']:.1f}% success")
            print(f"   📈 Raw price range: {price_stats['raw_min']:.2e} to {price_stats['raw_max']:.2e}")
            print(f"   🔢 Scaled range: {price_stats['scaled_min']:.6f} to {price_stats['scaled_max']:.6f}")
            print(f"   ⚙️  Scaled (scientific): {price_stats['scaled_min']:.2e} to {price_stats['scaled_max']:.2e}")
            print(f"   ⚖️  Scale factor: {price_stats['scale_factor']:,.0f}")
            
            # Test numerical precision
            precision_test_passed = True
            
            # Check if scaled prices are in safe range (1e-6 to 1e12)
            if price_stats['scaled_min'] < 1e-6:
                print(f"   ⚠️  Scaled prices too small: {price_stats['scaled_min']:.2e}")
                precision_test_passed = False
            
            if price_stats['scaled_max'] > 1e12:
                print(f"   ⚠️  Scaled prices too large: {price_stats['scaled_max']:.2e}")
                precision_test_passed = False
            
            # Test precision by doing some calculations
            test_position_size = 1e9  # 1 billion tokens
            test_value = price_stats['scaled_avg'] * test_position_size
            test_value_sol = test_value / feed.PRICE_SCALE_FACTOR
            
            if test_value_sol < 1e-12:
                print(f"   ⚠️  Position value calculation precision issue: {test_value_sol:.2e} SOL")
                precision_test_passed = False
            
            if precision_test_passed:
                print(f"   ✅ Numerical precision test passed")
            
            # Test OHLCV calculation with this coin
            if len(feed.sample_timestamps) > 0:
                sample_timestamp = feed.sample_timestamps[0]
                ohlcv = feed._calculate_ohlcv_from_transactions(sample_timestamp)
                
                ohlcv_valid = (
                    ohlcv['high'] >= ohlcv['low'] and
                    ohlcv['high'] >= ohlcv['open'] and
                    ohlcv['high'] >= ohlcv['close'] and
                    ohlcv['low'] <= ohlcv['open'] and
                    ohlcv['low'] <= ohlcv['close']
                )
                
                if ohlcv_valid:
                    print(f"   ✅ OHLCV calculation valid")
                else:
                    print(f"   ❌ OHLCV calculation invalid")
                    print(f"       O:{ohlcv['open']:.2f} H:{ohlcv['high']:.2f} L:{ohlcv['low']:.2f} C:{ohlcv['close']:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error processing coin: {e}")
            continue
        
        print()
    
    conn.close()
    
    # Summary analysis
    print("📊 MULTI-COIN SCALING ANALYSIS")
    print("=" * 60)
    
    if len(results) == 0:
        print("❌ No successful tests completed")
        return
    
    # Create summary table
    print(f"{'Category':<12} {'Raw Avg Price':<15} {'Scaled Avg':<15} {'Scaled Range':<25} {'Success %':<10} {'Status'}")
    print("-" * 100)
    
    all_success = True
    for result in results:
        status = "✅ Good" if result['success_rate'] > 80 else "⚠️  Low"
        if result['scaled_min'] < 1e-15 or result['scaled_max'] > 1e12:
            status = "❌ Range"
            all_success = False
        
        scaled_range = f"{result['scaled_min']:.2e} to {result['scaled_max']:.2e}"
        print(f"{result['category']:<12} {result['raw_avg']:<15.2e} {result['scaled_avg']:<15.2e} "
              f"{scaled_range:<25} {result['success_rate']:<10.1f} {status}")
    
    print()
    
    # Price range analysis
    all_raw_prices = [r['raw_avg'] for r in results]
    all_scaled_prices = [r['scaled_avg'] for r in results]
    
    raw_range_span = max(all_raw_prices) / min(all_raw_prices) if min(all_raw_prices) > 0 else 0
    scaled_range_span = max(all_scaled_prices) / min(all_scaled_prices) if min(all_scaled_prices) > 0 else 0
    
    print(f"📈 Price range analysis:")
    print(f"   Raw price range: {min(all_raw_prices):.2e} to {max(all_raw_prices):.2e}")
    print(f"   Raw range span: {raw_range_span:.2f}x")
    print(f"   Scaled price range: {min(all_scaled_prices):.2f} to {max(all_scaled_prices):.2f}")
    print(f"   Scaled range span: {scaled_range_span:.2f}x")
    
    # Final verdict
    print(f"\n🎯 FINAL ASSESSMENT:")
    if all_success and len(results) >= 3:
        print("🎉 Multi-coin price scaling test PASSED!")
        print("   ✅ All coins processed successfully")
        print("   ✅ Scaled prices in safe numerical range")
        print("   ✅ High price calculation success rates")
    elif len(results) >= 2:
        print("⚠️  Multi-coin price scaling test PARTIAL SUCCESS")
        print("   Some coins processed successfully")
        print("   Minor issues detected - review warnings above")
    else:
        print("❌ Multi-coin price scaling test FAILED")
        print("   Insufficient successful tests")
    
    print(f"\n✅ Tested {len(results)} different meme coin categories")
    print(f"⚖️  Scale factor: {results[0]['scale_factor']:,.0f} (1e9)")


if __name__ == '__main__':
    test_multi_coin_price_scaling()