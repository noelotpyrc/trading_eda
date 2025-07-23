#!/usr/bin/env python3
"""
Simple Example: How to use the backtesting framework
"""

import sys
sys.path.append('/Users/noel/projects/trading_eda')

from backtesting import (
    BacktestRunner,
    MLClassificationStrategy,
    create_sample_data,
    add_ml_signals
)


def simple_example():
    """Simple example showing basic usage"""
    print("=== SIMPLE BACKTESTING EXAMPLE ===")
    
    # 1. Create sample data
    print("1. Creating sample trading data...")
    data = create_sample_data(n_days=365, start_price=100.0)
    
    # 2. Add ML signals (using random signals for demo)
    print("2. Adding ML signals...")
    data = add_ml_signals(data, signal_source='random', seed=42)
    
    # 3. Set up backtest runner
    print("3. Setting up backtest runner...")
    runner = BacktestRunner(
        initial_cash=100000,  # Start with $100k
        commission=0.001      # 0.1% commission
    )
    
    # 4. Define strategy parameters
    strategy_params = {
        'ml_threshold': 0.6,      # Buy when ML signal > 0.6
        'stop_loss': 0.05,        # 5% stop loss
        'take_profit': 0.15,      # 15% take profit
        'min_confidence': 0.7     # Require 70% confidence
    }
    
    # 5. Run backtest
    print("4. Running backtest...")
    results = runner.run_backtest(
        data=data,
        strategy_class=MLClassificationStrategy,
        strategy_params=strategy_params
    )
    
    # 6. Show results
    print("\n5. Results:")
    print(f"   Initial Capital: ${results['initial_cash']:,.2f}")
    print(f"   Final Value: ${results['final_value']:,.2f}")
    print(f"   Total Return: {results['total_return']:.2f}%")
    print(f"   Profit/Loss: ${results['total_return_abs']:,.2f}")
    
    if 'trades' in results['analyzers']:
        trades = results['analyzers']['trades']
        print(f"   Total Trades: {trades.get('total_trades', 0)}")
        print(f"   Win Rate: {trades.get('win_rate', 0):.1f}%")
    
    print("\n✅ Backtest completed successfully!")
    
    return results


if __name__ == "__main__":
    simple_example()
