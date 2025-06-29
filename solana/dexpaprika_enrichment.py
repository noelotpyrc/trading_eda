#!/usr/bin/env python3
"""
DexPaprika API Enrichment Script
Enriches post-2024 meme coin data with metadata from DexPaprika API
"""

import pandas as pd
import requests
import json
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def get_token_info_dexpaprika(mint_address: str) -> Optional[Dict]:
    """
    Get token information from DexPaprika API
    
    Args:
        mint_address: Solana token mint address
        
    Returns:
        Dictionary with token metadata or None if not found
    """
    url = f"https://api.dexpaprika.com/networks/solana/tokens/{mint_address}"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            
            return {
                'mint_address': mint_address,
                'name': data.get('name', 'Unknown'),
                'symbol': data.get('symbol', 'UNKNOWN'),
                'total_supply': data.get('total_supply', 0),
                'price_usd': summary.get('price_usd', 0),
                'fdv': summary.get('fdv', 0),
                'liquidity_usd': summary.get('liquidity_usd', 0),
                'api_status': 'success'
            }
        else:
            return {
                'mint_address': mint_address,
                'api_status': f'error_{response.status_code}'
            }
            
    except Exception as e:
        return {
            'mint_address': mint_address,
            'api_status': f'error_{str(e)}'
        }

def enrich_coins_with_dexpaprika(input_csv: str, output_csv: str, max_tokens: int = None):
    """
    Enrich coin data with DexPaprika API information
    
    Args:
        input_csv: Path to input CSV with mint addresses
        output_csv: Path to output enriched CSV
        max_tokens: Maximum number of tokens to process (for testing)
    """
    
    print("🚀 DexPaprika API Enrichment Starting...")
    print("=" * 60)
    
    # Load coin addresses
    df = pd.read_csv(input_csv)
    print(f"📊 Loaded {len(df):,} coins from {input_csv}")
    
    # Limit for testing if specified
    if max_tokens:
        df = df.head(max_tokens)
        print(f"🔬 Processing first {max_tokens} coins for testing")
    
    # Initialize results list
    enriched_data = []
    
    # Process each token
    total_tokens = len(df)
    for i, row in df.iterrows():
        mint_address = row['mint_address']
        
        print(f"[{i+1:4d}/{total_tokens}] Processing {mint_address[:8]}...", end=" ")
        
        # Get token info from DexPaprika
        token_info = get_token_info_dexpaprika(mint_address)
        
        if token_info and token_info.get('api_status') == 'success':
            print(f"✅ {token_info.get('symbol', 'UNK')}")
        else:
            print("❌ No data")
        
        # Merge with original data
        enriched_row = {
            **row.to_dict(),  # Original trading data
            **token_info      # DexPaprika metadata
        }
        enriched_data.append(enriched_row)
        
        # Rate limiting to avoid API limits
        if i < total_tokens - 1:
            time.sleep(0.2)  # 200ms delay between requests
    
    # Save enriched data
    enriched_df = pd.DataFrame(enriched_data)
    enriched_df.to_csv(output_csv, index=False)
    
    # Summary
    successful = (enriched_df['api_status'] == 'success').sum()
    print(f"\n📊 ENRICHMENT SUMMARY:")
    print(f"   Total processed: {len(enriched_df):,}")
    print(f"   Successful API calls: {successful:,}")
    print(f"   Success rate: {successful/len(enriched_df):.1%}")
    print(f"   Saved to: {output_csv}")
    
    # Show preview of enriched data
    print(f"\n🏆 TOP ENRICHED TOKENS:")
    successful_tokens = enriched_df[enriched_df['api_status'] == 'success']
    if len(successful_tokens) > 0:
        for i, row in successful_tokens.head(10).iterrows():
            print(f"{i+1:2d}. {row['symbol']:<8} ({row['name'][:30]:<30}) | "
                  f"${row['price_usd']:>8.6f} | "
                  f"FDV: ${row['fdv']:>12,.0f} | "
                  f"Liquidity: ${row['liquidity_usd']:>12,.0f}")
    else:
        print("   No successful API calls to display")

if __name__ == '__main__':
    # Configuration
    INPUT_CSV = 'post2024_meme_coins_for_dexpaprika.csv'
    OUTPUT_CSV = 'post2024_meme_coins_enriched.csv'
    
    # For testing, start with just 50 tokens
    TEST_MODE = False
    MAX_TOKENS = 50 if TEST_MODE else None
    
    print("🎯 DexPaprika Enrichment Configuration:")
    print(f"   Input: {INPUT_CSV}")
    print(f"   Output: {OUTPUT_CSV}")
    print(f"   Test mode: {TEST_MODE}")
    if TEST_MODE:
        print(f"   Max tokens: {MAX_TOKENS}")
    print()
    
    # Run enrichment
    try:
        enrich_coins_with_dexpaprika(INPUT_CSV, OUTPUT_CSV)
        print("\n✅ Enrichment completed successfully!")
        print(f"💡 Next: Use {OUTPUT_CSV} for your trading signal analysis")
        
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_CSV} not found")
        print("💡 Run the coin selection notebook first to generate the CSV")
        
    except Exception as e:
        print(f"❌ Error during enrichment: {e}") 