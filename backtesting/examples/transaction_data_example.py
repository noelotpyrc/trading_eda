#!/usr/bin/env python3
"""
Transaction Data Example
Shows how to use cleaned Solana transaction data with proper separation of concerns
Data feed provides raw data, strategy handles intelligence internally
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append('/Users/noel/projects/trading_eda')
import os
import backtrader as bt
from backtesting.solana_transaction_feed import SolanaTransactionFeed
from backtesting.strategies import SimpleTransactionStrategy
from backtesting.onchain_broker import setup_onchain_broker


def create_sample_transaction_data(n_transactions=5000):
    """
    Create sample transaction data that mimics real Solana transactions
    Format matches your actual data structure from first_day_trades table
    """
    print(f"📊 Creating {n_transactions} sample Solana transactions...")
    
    # Generate transaction timestamps over 24 hours
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(days=1)
    
    # Create timestamps with realistic clustering (more activity during certain hours)
    np.random.seed(42)
    
    # Generate clustered timestamps (simulate realistic trading patterns)
    timestamps = []
    current_time = start_time
    
    while len(timestamps) < n_transactions and current_time < end_time:
        # Variable activity throughout the day
        hour = current_time.hour
        if 8 <= hour <= 22:  # More activity during "day" hours
            intensity = 2.0
        else:  # Lower activity at night
            intensity = 0.5
        
        # Random interval between transactions
        interval_minutes = np.random.exponential(scale=1.0 / intensity)
        current_time += timedelta(minutes=interval_minutes)
        
        if current_time < end_time:
            timestamps.append(current_time)
    
    # Trim to exact count
    timestamps = timestamps[:n_transactions]
    
    # Generate realistic transaction data
    transactions = []
    cumulative_volume = 0
    
    for i, timestamp in enumerate(timestamps):
        # Transaction size distribution (log-normal like real trading)
        sol_amount = np.random.lognormal(mean=0.5, sigma=1.5)  # 0.1 to 100+ SOL
        sol_amount = max(0.01, min(sol_amount, 500))  # Cap at reasonable range
        
        # Buy/sell ratio (slightly more buys to simulate growth)
        is_buy = np.random.choice([True, False], p=[0.55, 0.45])
        
        # Simulate trader addresses (random IDs)
        swapper = f"trader_{np.random.randint(1, 1000):04d}"
        
        # Simulate mint (token) address
        mint = f"token_{np.random.randint(1, 10):02d}_abc123def456"
        
        # Simulate prices (with some trending)
        base_price = 1.0 + (i / n_transactions) * 0.5  # Slight upward trend
        price_noise = np.random.normal(0, 0.05)
        price = max(0.1, base_price + price_noise)
        
        # Success rate (most transactions succeed)
        succeeded = np.random.choice([True, False], p=[0.95, 0.05])
        
        cumulative_volume += sol_amount
        
        # Create proper SOL trading format for price calculation
        if is_buy:  # SOL -> Token (buy token with SOL)
            swap_from_mint = 'So11111111111111111111111111111111111111112'  # SOL
            swap_to_mint = mint
            swap_from_amount = sol_amount
            swap_to_amount = sol_amount / price  # Token amount
        else:  # Token -> SOL (sell token for SOL)
            swap_from_mint = mint
            swap_to_mint = 'So11111111111111111111111111111111111111112'  # SOL
            swap_from_amount = sol_amount / price  # Token amount
            swap_to_amount = sol_amount
        
        transaction = {
            'block_timestamp': timestamp,
            'mint': mint,
            'succeeded': succeeded,
            'swapper': swapper,
            'sol_amount': sol_amount,
            'is_buy': is_buy,
            'price': price,
            'swap_from_amount': swap_from_amount,
            'swap_to_amount': swap_to_amount,
            'swap_from_mint': swap_from_mint,
            'swap_to_mint': swap_to_mint,
            'cumulative_volume': cumulative_volume
        }
        
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    
    print(f"✅ Generated transaction data:")
    print(f"   Transactions: {len(df)}")
    print(f"   Time span: {df['block_timestamp'].min()} to {df['block_timestamp'].max()}")
    print(f"   Buy transactions: {df['is_buy'].sum()} ({df['is_buy'].mean()*100:.1f}%)")
    print(f"   Total SOL volume: {df['sol_amount'].sum():.2f}")
    print(f"   Avg transaction size: {df['sol_amount'].mean():.4f} SOL")
    print(f"   Success rate: {df['succeeded'].mean()*100:.1f}%")
    
    return df


def test_transaction_data_strategy():
    """Test strategy using cleaned transaction data feed"""
    print("\n🔥 === TESTING CLEAN TRANSACTION STRATEGY ===")
    
    # Create sample transaction data
    transaction_df = create_sample_transaction_data(n_transactions=2000)
    
    # Setup Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add custom transaction data feed
    transaction_feed = SolanaTransactionFeed(
        transaction_data=transaction_df,
        aggregation_window=300,  # 5-minute windows
        min_transactions=10,     # Minimum 10 transactions per window
        datetime_col='block_timestamp',
        volume_col='sol_amount',
        direction_col='is_buy',
        trader_col='swapper',
        succeed_col='succeeded'
    )
    
    cerebro.adddata(transaction_feed)
    
    # Add simple transaction strategy (works with cleaned data feed)
    cerebro.addstrategy(
        SimpleTransactionStrategy,
        buy_ratio_threshold=0.6,        # Buy when buy ratio > 60%
        volume_threshold=5.0,           # Minimum 5 SOL volume
        trader_diversity_threshold=3,   # At least 3 unique traders
        position_size_pct=0.8,          # Use 80% of cash
        stop_loss=0.05,                 # 5% stop loss
        take_profit=0.12,               # 12% take profit
        verbose=True
    )
    
    # Setup onchain broker instead of traditional broker
    setup_onchain_broker(cerebro, initial_cash=100000)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f"💰 Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    
    # Run backtest
    print("🚀 Running transaction-level backtest...")
    results = cerebro.run()
    
    # Get results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - 100000) / 100000 * 100
    
    print(f"\n📈 === TRANSACTION STRATEGY RESULTS ===")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Detailed analysis
    strat = results[0]
    
    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    except:
        print("Sharpe Ratio: N/A")
    
    try:
        drawdown = strat.analyzers.drawdown.get_analysis()
        print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")
    except:
        print("Max Drawdown: N/A")
    
    try:
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        lost_trades = trades.get('lost', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades}")
        print(f"Losing Trades: {lost_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        if won_trades > 0:
            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            print(f"Average Win: ${avg_win:.2f}")
        
        if lost_trades > 0:
            avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
            print(f"Average Loss: ${avg_loss:.2f}")
            
    except Exception as e:
        print(f"Trade analysis error: {e}")
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'total_trades': trades.get('total', {}).get('total', 0) if 'trades' in locals() else 0
    }






def load_real_transaction_data():
    """
    Load real Solana transaction data if available
    """
    print("\n💾 === LOADING REAL TRANSACTION DATA ===")
    
    # Try to find real transaction data files
    potential_paths = [
        '/Volumes/Extreme SSD/trading_data/solana/first_day_trades',
        'data/solana_transactions.csv',
        'solana/data/first_day_trades.csv'
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    # Look for CSV files in directory
                    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
                    if csv_files:
                        file_path = os.path.join(path, csv_files[0])
                        print(f"📁 Loading from: {file_path}")
                        
                        # Load sample of data (for demo)
                        df = pd.read_csv(file_path, nrows=10000)  # First 10k transactions
                        
                        print(f"✅ Loaded {len(df)} real transactions")
                        print(f"   Columns: {list(df.columns)}")
                        
                        return df
                else:
                    print(f"📁 Loading from: {path}")
                    df = pd.read_csv(path, nrows=10000)
                    
                    print(f"✅ Loaded {len(df)} real transactions")
                    return df
                    
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
                continue
    
    print("⚠️ No real transaction data found, using simulated data")
    return None


if __name__ == "__main__":
    print("🚀 SOLANA TRANSACTION DATA BACKTESTING")
    print("=" * 50)
    
    # Try to load real data first
    real_data = load_real_transaction_data()
    
    if real_data is not None:
        print("🎯 Testing with REAL transaction data...")
        # Would need to adapt column names to match your actual data structure
        # test_with_real_data(real_data)
    
    # Run tests with simulated data
    print("🎯 Testing with SIMULATED transaction data...")
    
    tests = [
        test_transaction_data_strategy
    ]
    
    results = {}
    
    for test_func in tests:
        try:
            result = test_func()
            results[test_func.__name__] = result
            print(f"✅ {test_func.__name__} completed")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Transaction data testing completed!")
    print(f"📝 Key insight: Direct transaction processing enables more granular trading signals")