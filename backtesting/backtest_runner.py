#!/usr/bin/env python3
"""
Backtest Runner
Main backtesting engine with configurable strategies and analysis
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Type, List
import json
from datetime import datetime
import os

from .ml_strategy import MLClassificationStrategy, MLDataFeed, MultiSignalStrategy
from .base_strategy import BaseStrategy, SignalBasedStrategy
from .data_utils import prepare_data_for_backtest, validate_backtest_data


class BacktestRunner:
    """
    Main backtesting engine with configurable strategies and comprehensive analysis
    """
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 commission: float = 0.001,
                 verbose: bool = True):
        """
        Initialize backtest runner
        
        Args:
            initial_cash: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            verbose: Enable detailed logging
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.verbose = verbose
        self.results_history = []
    
    def run_backtest(self,
                     data: pd.DataFrame,
                     strategy_class: Type[BaseStrategy] = MLClassificationStrategy,
                     strategy_params: Optional[Dict[str, Any]] = None,
                     analyzers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a single backtest
        
        Args:
            data: DataFrame with OHLCV and signal data
            strategy_class: Strategy class to use
            strategy_params: Parameters for the strategy
            analyzers: List of analyzer names to add
            
        Returns:
            Dictionary with backtest results
        """
        if strategy_params is None:
            strategy_params = {}
        
        if analyzers is None:
            analyzers = ['sharpe', 'drawdown', 'returns', 'trades', 'positions']
        
        if self.verbose:
            print(f"\n=== RUNNING BACKTEST ===")
            print(f"Strategy: {strategy_class.__name__}")
            print(f"Initial Cash: ${self.initial_cash:,.2f}")
            print(f"Commission: {self.commission*100:.3f}%")
            print(f"Data Period: {data.index.min()} to {data.index.max()}")
            print(f"Total Bars: {len(data)}")
        
        # Validate data
        validation = validate_backtest_data(data)
        if not validation['valid']:
            raise ValueError(f"Data validation failed: {validation['errors']}")
        
        if validation['warnings'] and self.verbose:
            for warning in validation['warnings']:
                print(f"Warning: {warning}")
        
        # Create Cerebro engine
        cerebro = bt.Cerebro()
        
        # Add strategy with parameters
        cerebro.addstrategy(strategy_class, verbose=self.verbose, **strategy_params)
        
        # Create and add data feed
        data_feed = self._create_data_feed(data)
        cerebro.adddata(data_feed)
        
        # Set broker parameters
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        analyzer_instances = self._add_analyzers(cerebro, analyzers)
        
        # Run backtest
        if self.verbose:
            print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
        
        start_time = datetime.now()
        results = cerebro.run()
        end_time = datetime.now()
        
        strategy_result = results[0]
        final_value = cerebro.broker.getvalue()
        
        # Compile results
        backtest_results = {
            'strategy_name': strategy_class.__name__,
            'strategy_params': strategy_params,
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': (final_value - self.initial_cash) / self.initial_cash * 100,
            'total_return_abs': final_value - self.initial_cash,
            'commission_rate': self.commission,
            'run_time': (end_time - start_time).total_seconds(),
            'data_stats': validation['stats'],
            'analyzers': {}
        }
        
        # Extract analyzer results
        for analyzer_name in analyzers:
            analyzer_results = self._extract_analyzer_results(strategy_result, analyzer_name)
            if analyzer_results:
                backtest_results['analyzers'][analyzer_name] = analyzer_results
        
        # Store cerebro instance for plotting
        backtest_results['cerebro'] = cerebro
        backtest_results['strategy_result'] = strategy_result
        
        # Print summary
        if self.verbose:
            self._print_results_summary(backtest_results)
        
        # Store in history
        self.results_history.append(backtest_results)
        
        return backtest_results
    
    def run_parameter_optimization(self,
                                  data: pd.DataFrame,
                                  strategy_class: Type[BaseStrategy],
                                  param_grid: Dict[str, List[Any]],
                                  optimization_metric: str = 'total_return',
                                  max_combinations: int = 100) -> Dict[str, Any]:
        """
        Run parameter optimization using grid search
        
        Args:
            data: DataFrame with OHLCV and signal data
            strategy_class: Strategy class to optimize
            param_grid: Dictionary with parameter names and values to test
            optimization_metric: Metric to optimize
            max_combinations: Maximum number of parameter combinations to test
            
        Returns:
            Dictionary with optimization results
        """
        from itertools import product
        
        print(f"\n=== PARAMETER OPTIMIZATION ===")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Optimization metric: {optimization_metric}")
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        if len(combinations) > max_combinations:
            print(f"Warning: {len(combinations)} combinations, limiting to {max_combinations}")
            combinations = combinations[:max_combinations]
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        optimization_results = []
        best_result = None
        best_metric = float('-inf')
        
        for i, param_combination in enumerate(combinations, 1):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combination))
            
            try:
                # Run backtest with current parameters
                result = self.run_backtest(
                    data=data,
                    strategy_class=strategy_class,
                    strategy_params=params,
                    analyzers=['returns', 'drawdown', 'trades']
                )
                
                # Extract metric value
                if optimization_metric in result:
                    metric_value = result[optimization_metric]
                elif optimization_metric in result.get('analyzers', {}):
                    metric_value = result['analyzers'][optimization_metric]
                else:
                    print(f"Warning: Metric {optimization_metric} not found in results")
                    metric_value = 0
                
                # Store result
                optimization_result = {
                    'params': params,
                    'metric_value': metric_value,
                    'full_result': result
                }
                optimization_results.append(optimization_result)
                
                # Track best result
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_result = optimization_result
                
                if self.verbose and i % 10 == 0:
                    print(f"  Completed {i}/{len(combinations)} combinations")
                
            except Exception as e:
                print(f"  Error with params {params}: {e}")
                continue
        
        # Compile optimization summary
        optimization_summary = {
            'strategy_class': strategy_class.__name__,
            'optimization_metric': optimization_metric,
            'total_combinations': len(combinations),
            'successful_runs': len(optimization_results),
            'best_params': best_result['params'] if best_result else None,
            'best_metric_value': best_metric,
            'best_result': best_result,
            'all_results': optimization_results
        }
        
        if self.verbose:
            print(f"\nOptimization completed!")
            print(f"Best {optimization_metric}: {best_metric:.4f}")
            print(f"Best parameters: {best_result['params'] if best_result else 'None'}")
        
        return optimization_summary
    
    def compare_strategies(self,
                          data: pd.DataFrame,
                          strategy_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data
        
        Args:
            data: DataFrame with OHLCV and signal data
            strategy_configs: List of strategy configurations
            
        Returns:
            DataFrame with comparison results
        """
        print(f"\n=== STRATEGY COMPARISON ===")
        print(f"Comparing {len(strategy_configs)} strategies...")
        
        comparison_results = []
        
        for i, config in enumerate(strategy_configs, 1):
            strategy_class = config.get('strategy_class', MLClassificationStrategy)
            strategy_params = config.get('params', {})
            strategy_name = config.get('name', f"Strategy_{i}")
            
            print(f"\nRunning {strategy_name}...")
            
            try:
                result = self.run_backtest(
                    data=data,
                    strategy_class=strategy_class,
                    strategy_params=strategy_params
                )
                
                # Extract key metrics for comparison
                comparison_result = {
                    'strategy_name': strategy_name,
                    'strategy_class': strategy_class.__name__,
                    'total_return_pct': result['total_return'],
                    'final_value': result['final_value'],
                    'total_return_abs': result['total_return_abs'],
                }
                
                # Add analyzer metrics
                analyzers = result.get('analyzers', {})
                if 'sharpe' in analyzers:
                    comparison_result['sharpe_ratio'] = analyzers['sharpe'].get('sharperatio', 0)
                if 'drawdown' in analyzers:
                    comparison_result['max_drawdown'] = analyzers['drawdown'].get('max_drawdown', 0)
                if 'trades' in analyzers:
                    trades = analyzers['trades']
                    comparison_result['total_trades'] = trades.get('total_trades', 0)
                    comparison_result['win_rate'] = trades.get('win_rate', 0)
                
                comparison_results.append(comparison_result)
                
            except Exception as e:
                print(f"Error running {strategy_name}: {e}")
                continue
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            # Sort by total return
            comparison_df = comparison_df.sort_values('total_return_pct', ascending=False)
            
            if self.verbose:
                print(f"\n=== STRATEGY COMPARISON RESULTS ===")
                print(comparison_df.round(4))
        
        return comparison_df
    
    def _create_data_feed(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """Create appropriate data feed based on available columns"""
        # Check if we have ML signal columns
        if 'ml_signal' in data.columns and 'ml_confidence' in data.columns:
            return MLDataFeed(
                dataname=data,
                ml_signal=data.columns.get_loc('ml_signal'),
                ml_confidence=data.columns.get_loc('ml_confidence')
            )
        else:
            return bt.feeds.PandasData(dataname=data)
    
    def _add_analyzers(self, cerebro: bt.Cerebro, analyzer_names: List[str]) -> Dict[str, Any]:
        """Add analyzers to cerebro"""
        analyzer_map = {
            'sharpe': bt.analyzers.SharpeRatio,
            'drawdown': bt.analyzers.DrawDown,
            'returns': bt.analyzers.Returns,
            'trades': bt.analyzers.TradeAnalyzer,
            'positions': bt.analyzers.PositionsValue,
            'transactions': bt.analyzers.Transactions,
            'calmar': bt.analyzers.CalmarRatio,
            'vwr': bt.analyzers.VWR,  # Variability-Weighted Return
        }
        
        added_analyzers = {}
        for analyzer_name in analyzer_names:
            if analyzer_name in analyzer_map:
                cerebro.addanalyzer(analyzer_map[analyzer_name], _name=analyzer_name)
                added_analyzers[analyzer_name] = analyzer_map[analyzer_name]
            else:
                print(f"Warning: Unknown analyzer: {analyzer_name}")
        
        return added_analyzers
    
    def _extract_analyzer_results(self, strategy_result: Any, analyzer_name: str) -> Dict[str, Any]:
        """Extract results from a specific analyzer"""
        try:
            analyzer = getattr(strategy_result.analyzers, analyzer_name)
            analysis = analyzer.get_analysis()
            
            # Process specific analyzer types
            if analyzer_name == 'sharpe':
                return {'sharperatio': analysis.get('sharperatio', 0)}
            
            elif analyzer_name == 'drawdown':
                return {
                    'max_drawdown': analysis.get('max', {}).get('drawdown', 0),
                    'max_drawdown_period': analysis.get('max', {}).get('len', 0)
                }
            
            elif analyzer_name == 'trades':
                total = analysis.get('total', {})
                won = analysis.get('won', {})
                lost = analysis.get('lost', {})
                
                return {
                    'total_trades': total.get('total', 0),
                    'win_rate': (won.get('total', 0) / total.get('total', 1)) * 100 if total.get('total', 0) > 0 else 0,
                    'avg_win': won.get('pnl', {}).get('average', 0),
                    'avg_loss': lost.get('pnl', {}).get('average', 0),
                    'profit_factor': abs(won.get('pnl', {}).get('total', 0) / lost.get('pnl', {}).get('total', 1)) if lost.get('pnl', {}).get('total', 0) != 0 else float('inf')
                }
            
            else:
                return analysis
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not extract {analyzer_name} results: {e}")
            return {}
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print summary of backtest results"""
        print(f"\nFinal Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Absolute Return: ${results['total_return_abs']:,.2f}")
        
        analyzers = results.get('analyzers', {})
        
        if 'sharpe' in analyzers:
            sharpe = analyzers['sharpe'].get('sharperatio', 0)
            print(f"Sharpe Ratio: {sharpe:.3f}")
        
        if 'drawdown' in analyzers:
            max_dd = analyzers['drawdown'].get('max_drawdown', 0)
            print(f"Max Drawdown: {max_dd:.2f}%")
        
        if 'trades' in analyzers:
            trades = analyzers['trades']
            total_trades = trades.get('total_trades', 0)
            win_rate = trades.get('win_rate', 0)
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.1f}%")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save backtest results to file"""
        # Remove non-serializable objects
        serializable_results = results.copy()
        serializable_results.pop('cerebro', None)
        serializable_results.pop('strategy_result', None)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
    
    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot backtest results"""
        try:
            cerebro = results.get('cerebro')
            if cerebro:
                fig = cerebro.plot(style='candlestick', barup='green', bardown='red')
                if save_path:
                    fig[0][0].savefig(save_path)
                    print(f"Plot saved to: {save_path}")
            else:
                print("No cerebro instance available for plotting")
        except Exception as e:
            print(f"Plotting not available: {e}")


def quick_backtest(data_file: str,
                   strategy_class: Type[BaseStrategy] = MLClassificationStrategy,
                   strategy_params: Optional[Dict[str, Any]] = None,
                   signal_source: str = 'random',
                   **kwargs) -> Dict[str, Any]:
    """
    Quick backtest function for simple use cases
    
    Args:
        data_file: Path to OHLCV data file
        strategy_class: Strategy class to use
        strategy_params: Strategy parameters
        signal_source: Source of ML signals ('inference', 'random', 'file')
        **kwargs: Additional parameters
        
    Returns:
        Backtest results
    """
    # Prepare data
    data = prepare_data_for_backtest(
        data_file=data_file,
        signal_source=signal_source,
        **kwargs
    )
    
    # Run backtest
    runner = BacktestRunner()
    results = runner.run_backtest(
        data=data,
        strategy_class=strategy_class,
        strategy_params=strategy_params or {}
    )
    
    return results
