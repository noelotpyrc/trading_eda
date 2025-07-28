#!/usr/bin/env python3
"""
Transaction Diagnosis Example
Demonstrates how to diagnose trading strategy decisions with cleaned transaction data
Uses DiagnosticSimpleStrategy which shows proper separation of concerns:
- Data feed provides raw transaction data only
- Strategy calculates indicators internally and provides detailed logging
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append('/Users/noel/projects/trading_eda')
import os
import backtrader as bt
from backtesting.solana_transaction_feed import SolanaTransactionFeed
from backtesting.strategies import DiagnosticSimpleStrategy
from backtesting.onchain_broker import setup_onchain_broker


def create_sample_transaction_data(n_transactions=5000):
    """
    Create sample transaction data that mimics real Solana transactions
    
    This generates synthetic but realistic transaction patterns including:
    - Price trends and volatility
    - Buy/sell pressure cycles
    - Whale activity periods
    - Volume fluctuations
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = []
    current_time = start_time
    
    for i in range(n_transactions):
        # Add random time intervals (1-30 seconds between transactions)
        current_time += timedelta(seconds=np.random.exponential(5))
        timestamps.append(current_time)
    
    # Generate price data with trend and volatility
    base_price = 100.0
    trend = 0.0001  # Slight upward trend
    volatility = 0.02
    
    prices = []
    current_price = base_price
    
    for i in range(n_transactions):
        # Add trend and random walk
        price_change = trend + np.random.normal(0, volatility)
        current_price *= (1 + price_change)
        
        # Add some mean reversion to keep price reasonable
        if current_price > base_price * 1.5:
            current_price *= 0.99
        elif current_price < base_price * 0.5:
            current_price *= 1.01
            
        prices.append(current_price)
    
    # Generate transaction volumes (SOL amounts)
    # Most transactions are small, but with occasional large ones
    volumes = np.random.lognormal(mean=1, sigma=1.5, size=n_transactions)
    volumes = np.clip(volumes, 0.1, 1000)  # Keep reasonable bounds
    
    # Generate buy/sell signals with some market cycles
    # Create periods of buying pressure and selling pressure
    buy_probabilities = []
    cycle_length = 200  # Transactions per cycle
    
    for i in range(n_transactions):
        cycle_position = (i % cycle_length) / cycle_length
        # Create sine wave for buy pressure cycles
        base_buy_prob = 0.5 + 0.2 * np.sin(2 * np.pi * cycle_position)
        # Add noise
        noise = np.random.normal(0, 0.1)
        buy_prob = np.clip(base_buy_prob + noise, 0.1, 0.9)
        buy_probabilities.append(buy_prob)
    
    is_buy = np.random.binomial(1, buy_probabilities)
    
    # Generate trader addresses (hash-like strings)
    trader_addresses = []
    for i in range(n_transactions):
        # Some traders make multiple transactions, others are one-time
        if np.random.random() < 0.3:  # 30% chance of repeat trader
            if trader_addresses and np.random.random() < 0.5:
                # Reuse recent trader
                recent_traders = trader_addresses[-20:] if len(trader_addresses) >= 20 else trader_addresses
                trader_addresses.append(np.random.choice(recent_traders))
            else:
                # New trader
                trader_hash = f"trader_{np.random.randint(1000000, 9999999):07d}"
                trader_addresses.append(trader_hash)
        else:
            trader_hash = f"trader_{np.random.randint(1000000, 9999999):07d}"
            trader_addresses.append(trader_hash)
    
    # Generate transaction size categories based on volume
    categories = []
    for vol in volumes:
        if vol <= 1:
            categories.append(0)  # Small
        elif vol <= 10:
            categories.append(1)  # Medium
        elif vol <= 100:
            categories.append(2)  # Big
        else:
            categories.append(3)  # Whale
    
    # Create success field (most transactions succeed)
    succeeded = np.random.choice([True, False], size=n_transactions, p=[0.95, 0.05])
    
    # Generate token mint addresses
    token_mints = []
    available_tokens = [f"token_{i:02d}_{''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), 6))}" 
                       for i in range(10)]
    
    for i in range(n_transactions):
        # Most transactions are for popular tokens
        if np.random.random() < 0.8:
            mint = np.random.choice(available_tokens[:5])  # Popular tokens
        else:
            mint = np.random.choice(available_tokens)  # Any token
        token_mints.append(mint)
    
    # Create DataFrame with required SOL trading format columns
    df_data = []
    for i in range(n_transactions):
        # Create proper SOL trading format for price calculation
        if is_buy[i]:  # SOL -> Token (buy token with SOL)
            swap_from_mint = 'So11111111111111111111111111111111111111112'  # SOL
            swap_to_mint = token_mints[i]
            swap_from_amount = volumes[i]
            swap_to_amount = volumes[i] / prices[i]  # Token amount
        else:  # Token -> SOL (sell token for SOL)
            swap_from_mint = token_mints[i]
            swap_to_mint = 'So11111111111111111111111111111111111111112'  # SOL
            swap_from_amount = volumes[i] / prices[i]  # Token amount
            swap_to_amount = volumes[i]
        
        transaction = {
            'block_timestamp': timestamps[i],
            'price': prices[i],
            'sol_amount': volumes[i],
            'is_buy': is_buy[i],
            'swapper': trader_addresses[i],
            'trader_hash': trader_addresses[i],  # Duplicate for compatibility
            'txn_size_category': categories[i],
            'succeeded': succeeded[i],
            'mint': token_mints[i],
            'swap_from_mint': swap_from_mint,
            'swap_to_mint': swap_to_mint,
            'swap_from_amount': swap_from_amount,
            'swap_to_amount': swap_to_amount
        }
        df_data.append(transaction)
    
    df = pd.DataFrame(df_data)
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('block_timestamp').reset_index(drop=True)
    
    print(f"📊 Generated {len(df)} sample transactions")
    print(f"   Time range: {df['block_timestamp'].min()} to {df['block_timestamp'].max()}")
    print(f"   Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
    print(f"   Buy ratio: {df['is_buy'].mean():.1%}")
    print(f"   Volume range: {df['sol_amount'].min():.2f} to {df['sol_amount'].max():.2f} SOL")
    print(f"   Unique traders: {df['swapper'].nunique()}")
    print(f"   Success rate: {df['succeeded'].mean():.1%}")
    
    return df


def run_diagnostic_backtest():
    """
    Run a diagnostic backtest that provides comprehensive analysis
    of trading decisions, market conditions, and strategy performance.
    """
    print("🚀 RUNNING DIAGNOSTIC SOLANA TRANSACTION BACKTEST")
    print("="*60)
    
    # Create sample transaction data
    transaction_data = create_sample_transaction_data(3000)
    
    # Initialize Backtrader
    cerebro = bt.Cerebro()
    
    # Set up the onchain broker for transaction-level execution
    setup_onchain_broker(cerebro, initial_cash=100000)
    
    # Add transaction data feed
    data_feed = SolanaTransactionFeed(
        transaction_data=transaction_data,
        aggregation_window=120,    # 2-minute windows
        min_transactions=8,        # Need at least 8 transactions per window
    )
    cerebro.adddata(data_feed)
    
    # Add diagnostic strategy with detailed parameters (works with cleaned data feed)
    cerebro.addstrategy(
        DiagnosticSimpleStrategy,
        # Trading thresholds
        buy_ratio_threshold=0.58,           # Slightly more sensitive
        volume_threshold=4.0,               # Lower volume threshold
        trader_diversity_threshold=2,       # At least 2 unique traders
        
        # Risk management
        position_size_pct=0.85,             # Use 85% of cash for positions
        stop_loss=0.04,                     # 4% stop loss
        take_profit=0.12,                   # 12% take profit
        lookback_periods=3,                 # 3-period momentum calculation
        
        # Diagnostic settings
        verbose=True,                       # Enable detailed logging
    )
    
    # Print initial conditions
    print(f"\n📋 BACKTEST CONFIGURATION:")
    print(f"   Initial Cash: ${cerebro.broker.get_cash():,.2f}")
    print(f"   Commission: {cerebro.broker.getcommissioninfo(data_feed).p.commission*100:.3f}%")
    print(f"   Data Points: {len(transaction_data)}")
    print(f"   Date Range: {transaction_data['block_timestamp'].min()} to {transaction_data['block_timestamp'].max()}")
    
    # Run the backtest
    print(f"\n🏃 RUNNING DIAGNOSTIC BACKTEST...")
    print("-" * 60)
    
    results = cerebro.run()
    strategy = results[0]
    
    # Final results
    final_value = cerebro.broker.get_value()
    total_return = (final_value - 100000) / 100000 * 100
    
    print(f"\n" + "="*60)
    print(f"📈 FINAL RESULTS")
    print(f"="*60)
    print(f"Initial Portfolio Value: $100,000.00")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"="*60)
    
    return strategy, transaction_data


if __name__ == "__main__":
    print("Starting Solana Transaction Diagnostic Backtest...")
    
    # Run the diagnostic backtest
    strategy_result, data = run_diagnostic_backtest()
    
    print(f"\n✅ Diagnostic backtest completed!")
    print(f"📊 Review the detailed logs above for comprehensive analysis")
    print(f"🔍 The DiagnosticSolanaStrategy provides:")
    print(f"   - Market condition tracking every period")
    print(f"   - Signal analysis and decision logging") 
    print(f"   - Trade execution details and PnL tracking")
    print(f"   - Comprehensive performance diagnostics")