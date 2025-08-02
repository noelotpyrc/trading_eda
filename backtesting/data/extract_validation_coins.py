#!/usr/bin/env python3
"""
Extract Validation Coin IDs
"""

import pickle
import pandas as pd

# Load validation dataset
val_set_path = '/Volumes/Extreme SSD/trading_data/solana/models/classification_forward/datasets/val_set.pkl'

with open(val_set_path, 'rb') as f:
    val_data = pickle.load(f)

# Extract metadata
metadata_val = val_data['metadata_val']

# Create summary by coin_id
coin_summary = metadata_val.groupby('coin_id').agg({
    'total_transactions': 'sum',  # Sum total transactions per coin
    'coin_id': 'count'  # Count of samples per coin
}).rename(columns={'coin_id': 'sample_count'}).reset_index()

# Save to CSV
output_file = 'validation_coins_summary.csv'
coin_summary.to_csv(output_file, index=False)

print(f"Found {len(coin_summary)} unique coin IDs")
print(f"Summary saved to: {output_file}")
print(f"\nTop 10 coins by total transactions:")
print(coin_summary.nlargest(10, 'total_transactions')[['coin_id', 'total_transactions', 'sample_count']].to_string(index=False))