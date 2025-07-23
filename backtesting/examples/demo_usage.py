#!/usr/bin/env python3
"""
Demo Usage Examples for Backtesting Framework
Shows how to use the modular backtesting framework
"""

import sys
import os

# Add project root to path
sys.path.append('/Users/noel/projects/trading_eda')

from backtesting import (
    BacktestRunner,
    MLClassificationStrategy,
    MultiSignalStrategy,
    quick_backtest,
    create_sample_data,
    prepare_data_for_backtest
)


def demo_basic_backtest():
    """Demo 1: Basic ML classification strategy backtest"""
    print("=== DEMO 1: BASIC ML CLASSIFICATION BACKTEST ===")
    
    # Create sample data
    data = create_sample_data(n_days=365, start_price=100.0, volatility=0.02)
    
    # Add random ML signals for demo
    from backtesting.data_utils import add_ml_signals
    data = add_ml_signals(data, signal_source='random', seed=42)
    
    # Run backtest
    runner = BacktestRunner(initial_cash=100000, commission=0.001)
    
    strategy_params = {
        'ml_threshold': 0.65,
        'stop_loss': 0.08,
        'take_profit': 0.20,
        'min_confidence': 0.6
    }
    
    results = runner.run_backtest(
        data=data,
        strategy_class=MLClassificationStrategy,
        strategy_params=strategy_params
    )
    
    print(f"✅ Basic backtest completed!")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    
    return results


def demo_parameter_optimization():
    """Demo 2: Parameter optimization"""
    print("\n=== DEMO 2: PARAMETER OPTIMIZATION ===")
    
    # Create sample data
    data = create_sample_data(n_days=200, volatility=0.025)
    from backtesting.data_utils import add_ml_signals
    data = add_ml_signals(data, signal_source='random', seed=123)
    
    # Define parameter grid
    param_grid = {
        'ml_threshold': [0.55, 0.6, 0.65, 0.7],
        'stop_loss': [0.05, 0.08, 0.10],
        'take_profit': [0.15, 0.20, 0.25],
        'min_confidence': [0.6, 0.7, 0.8]
    }
    
    # Run optimization
    runner = BacktestRunner(initial_cash=50000, verbose=False)
    
    optimization_results = runner.run_parameter_optimization(
        data=data,
        strategy_class=MLClassificationStrategy,
        param_grid=param_grid,
        optimization_metric='total_return',
        max_combinations=20  # Limit for demo
    )
    
    print(f"✅ Parameter optimization completed!")
    print(f"Best parameters: {optimization_results['best_params']}")
    print(f"Best return: {optimization_results['best_metric_value']:.2f}%")
    
    return optimization_results


def demo_strategy_comparison():
    """Demo 3: Strategy comparison"""
    print("\n=== DEMO 3: STRATEGY COMPARISON ===")
    
    # Create sample data
    data = create_sample_data(n_days=300, volatility=0.02)
    from backtesting.data_utils import add_ml_signals
    data = add_ml_signals(data, signal_source='random', seed=456)
    
    # Define strategy configurations
    strategy_configs = [
        {
            'name': 'Conservative ML',
            'strategy_class': MLClassificationStrategy,
            'params': {
                'ml_threshold': 0.7,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'min_confidence': 0.8
            }
        },
        {
            'name': 'Aggressive ML',
            'strategy_class': MLClassificationStrategy,
            'params': {
                'ml_threshold': 0.55,
                'stop_loss': 0.10,
                'take_profit': 0.25,
                'min_confidence': 0.6
            }
        },
        {
            'name': 'Multi-Signal',
            'strategy_class': MultiSignalStrategy,
            'params': {
                'signal_weights': {'ml': 0.7, 'technical': 0.3},
                'ensemble_threshold': 0.6,
                'min_confidence': 0.7
            }
        }
    ]
    
    # Run comparison
    runner = BacktestRunner(initial_cash=75000, verbose=False)
    
    comparison_df = runner.compare_strategies(
        data=data,
        strategy_configs=strategy_configs
    )
    
    print(f"✅ Strategy comparison completed!")
    print("\nResults:")
    print(comparison_df[['strategy_name', 'total_return_pct', 'max_drawdown', 'win_rate']].round(2))
    
    return comparison_df


def demo_different_signal_sources():
    """Demo 4: Different signal sources"""
    print("\n=== DEMO 4: DIFFERENT SIGNAL SOURCES ===")
    
    # Create base data
    data = create_sample_data(n_days=250)
    
    signal_sources = ['random', 'technical']
    results = {}
    
    for source in signal_sources:
        print(f"\nTesting {source} signals...")
        
        # Prepare data with specific signal source
        if source == 'technical':
            # Use technical indicator signals
            from backtesting.signal_generators import TechnicalIndicatorSignalGenerator
            
            # Add technical indicators
            from backtesting.data_utils import add_technical_indicators
            data_with_signals = add_technical_indicators(data.copy())
            
            # Create mock signals based on technical indicators
            data_with_signals['ml_signal'] = ((data_with_signals['close'] > data_with_signals['sma_10']) & 
                                            (data_with_signals['rsi'] < 70)).astype(float) * 0.7 + 0.3
            data_with_signals['ml_confidence'] = 0.75
        else:
            # Use random signals
            from backtesting.data_utils import add_ml_signals
            data_with_signals = add_ml_signals(data.copy(), signal_source='random')
        
        # Run backtest
        runner = BacktestRunner(initial_cash=50000, verbose=False)
        result = runner.run_backtest(
            data=data_with_signals,
            strategy_class=MLClassificationStrategy,
            strategy_params={'ml_threshold': 0.6}
        )
        
        results[source] = result
        print(f"  Return: {result['total_return']:.2f}%")
    
    print(f"\n✅ Signal source comparison completed!")
    
    return results


def demo_quick_backtest():
    """Demo 5: Quick backtest function"""
    print("\n=== DEMO 5: QUICK BACKTEST FUNCTION ===")
    
    # Create temporary data file
    sample_data = create_sample_data(n_days=180)
    temp_file = '/tmp/sample_trading_data.csv'
    sample_data.to_csv(temp_file)
    
    print(f"Created temporary data file: {temp_file}")
    
    # Use quick backtest function
    results = quick_backtest(
        data_file=temp_file,
        strategy_class=MLClassificationStrategy,
        strategy_params={
            'ml_threshold': 0.6,
            'stop_loss': 0.08,
            'take_profit': 0.18
        },
        signal_source='random',
        seed=789
    )
    
    print(f"✅ Quick backtest completed!")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    
    # Clean up
    os.remove(temp_file)
    
    return results


def demo_custom_strategy():
    """Demo 6: Creating a custom strategy"""
    print("\n=== DEMO 6: CUSTOM STRATEGY ===")
    
    from backtesting.base_strategy import BaseStrategy
    import backtrader as bt
    
    class SimpleMovingAverageStrategy(BaseStrategy):
        """Simple moving average crossover strategy"""
        
        params = (
            ('sma_short_period', 10),
            ('sma_long_period', 30),
        )
        
        def _initialize_strategy(self):
            """Initialize moving averages"""
            self.sma_short = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.sma_short_period
            )
            self.sma_long = bt.indicators.SimpleMovingAverage(
                self.datas[0], period=self.params.sma_long_period
            )
        
        def _get_signal(self):
            """Generate signal based on moving average crossover"""
            if (len(self.datas[0]) < max(self.params.sma_short_period, self.params.sma_long_period)):
                return 0.5, 0.5
            
            # Buy signal when short MA > long MA
            if self.sma_short[0] > self.sma_long[0]:
                signal = 0.8  # Strong buy
            else:
                signal = 0.2  # Strong sell
            
            confidence = 0.7
            return signal, confidence
        
        def _should_buy(self, signal_strength, confidence):
            """Buy when signal is strong"""
            return signal_strength > 0.7
    
    # Test custom strategy
    data = create_sample_data(n_days=200, trend=0.001)  # Add upward trend
    
    runner = BacktestRunner(initial_cash=50000, verbose=False)
    results = runner.run_backtest(
        data=data,
        strategy_class=SimpleMovingAverageStrategy,
        strategy_params={
            'sma_short_period': 10,
            'sma_long_period': 20
        }
    )
    
    print(f"✅ Custom strategy backtest completed!")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    
    return results


def run_all_demos():
    """Run all demo examples"""
    print("🚀 BACKTESTING FRAMEWORK DEMO\n")
    
    demos = [
        demo_basic_backtest,
        demo_parameter_optimization,
        demo_strategy_comparison,
        demo_different_signal_sources,
        demo_quick_backtest,
        demo_custom_strategy
    ]
    
    results = {}
    
    for demo_func in demos:
        try:
            result = demo_func()
            results[demo_func.__name__] = result
        except Exception as e:
            print(f"❌ {demo_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Demo completed! {len(results)}/{len(demos)} demos successful.")
    
    return results


if __name__ == "__main__":
    run_all_demos()
