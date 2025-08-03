#!/usr/bin/env python3
"""
Batch convert all validation coins to OHLVC signals format
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/Users/noel/projects/trading_eda')

from convert_to_ohlvc_signals import OHLVCSignalConverter

def main():
    print("🚀 Starting batch conversion of validation coins")
    
    # Load validation coins
    csv_path = 'validation_coins_summary.csv'
    df = pd.read_csv(csv_path)
    coin_ids = df['coin_id'].tolist()
    
    print(f"📊 Found {len(coin_ids)} validation coins to convert")
    
    # Initialize converter
    converter = OHLVCSignalConverter()
    
    # Convert all coins
    results = converter.convert_multiple_coins(coin_ids, candle_interval_minutes=1)
    
    # Generate summary report
    converter.create_summary_report(results)
    
    # Save results summary
    successful_coins = [coin_id for coin_id, path in results.items() if path is not None]
    failed_coins = [coin_id for coin_id, path in results.items() if path is None]
    
    # Save successful conversions list
    if successful_coins:
        success_df = pd.DataFrame({'coin_id': successful_coins})
        success_df.to_csv('successful_conversions.csv', index=False)
        print(f"✅ Saved {len(successful_coins)} successful conversions to successful_conversions.csv")
    
    # Save failed conversions list
    if failed_coins:
        failed_df = pd.DataFrame({'coin_id': failed_coins})
        failed_df.to_csv('failed_conversions.csv', index=False)
        print(f"❌ Saved {len(failed_coins)} failed conversions to failed_conversions.csv")
    
    print(f"\n🎉 Batch conversion completed!")
    print(f"✅ Success rate: {len(successful_coins)}/{len(coin_ids)} ({len(successful_coins)/len(coin_ids)*100:.1f}%)")

if __name__ == "__main__":
    main()