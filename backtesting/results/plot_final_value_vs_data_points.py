#!/usr/bin/env python3
"""
Plot final_value vs data_points from batch summary results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the batch summary data
batch_id = 'batch_20250806_225355'
df = pd.read_csv(f'backtesting/results/{batch_id}/batch_summary.csv')

print(f"Loaded {len(df)} records")
print(f"Data points range: {df['data_points'].min()} - {df['data_points'].max()}")
print(f"Final value range: {df['final_value'].min():.2f} - {df['final_value'].max():.2f}")

# Create the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(df['data_points'], df['final_value'], alpha=0.6, s=20)

# Add a horizontal line at initial value (100)
plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Initial Value (100)')

# Log scale for y-axis to better see the distribution
plt.yscale('log')

# Labels and title
plt.xlabel('Data Points (Number of 1-minute bars)')
plt.ylabel('Final Portfolio Value (Log Scale)')
plt.title(f'Final Portfolio Value vs Data Points\nBatch Results: {batch_id}')
plt.grid(True, alpha=0.3)
plt.legend()

# Add some statistics
profitable_coins = df[df['final_value'] > 100]
print(f"\nProfitable coins: {len(profitable_coins)}/{len(df)} ({len(profitable_coins)/len(df)*100:.1f}%)")
print(f"Best performer: {df.loc[df['final_value'].idxmax(), 'coin_id']} with {df['final_value'].max():.2f}")

# Color code points above/below 100
colors = ['green' if x > 100 else 'red' for x in df['final_value']]
plt.scatter(df['data_points'], df['final_value'], c=colors, alpha=0.6, s=20)

# Save and show
plt.tight_layout()
plt.savefig('final_value_vs_data_points.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as 'final_value_vs_data_points.png'")