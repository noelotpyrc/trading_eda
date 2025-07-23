# Modular Backtesting Framework

A flexible, modular backtesting framework built on Backtrader for testing various trading strategies with machine learning signals.

## Structure

```
backtesting/
├── __init__.py              # Main package exports
├── base_strategy.py         # Abstract base strategy classes
├── ml_strategy.py          # ML classification strategy implementation
├── signal_generators.py    # Pluggable signal generation components
├── data_utils.py           # Data loading and preprocessing utilities
├── backtest_runner.py      # Main backtesting engine
├── examples/
│   ├── simple_example.py   # Basic usage example
│   └── demo_usage.py       # Comprehensive demo with all features
└── README.md               # This file
```

## Key Features

### 🎯 **Modular Design**
- **Base Strategy Classes**: Abstract framework for any trading strategy
- **Signal Generators**: Pluggable components for different signal sources
- **Data Utilities**: Flexible data loading and preprocessing
- **Backtest Runner**: Comprehensive backtesting engine with analysis

### 🤖 **ML Integration**
- **Classification Strategies**: Built-in support for ML classification models
- **Multiple Signal Sources**: ML inference, technical indicators, random, pre-computed
- **Ensemble Strategies**: Combine multiple signal sources with custom weights

### 📊 **Advanced Analysis**
- **Parameter Optimization**: Grid search optimization with any metric
- **Strategy Comparison**: Compare multiple strategies on same data
- **Comprehensive Metrics**: Sharpe ratio, drawdown, win rate, profit factor, etc.
- **Result Visualization**: Built-in plotting capabilities

## Quick Start

### Basic Usage

```python
from backtesting import (
    BacktestRunner,
    MLClassificationStrategy,
    create_sample_data,
    add_ml_signals
)

# Create sample data
data = create_sample_data(n_days=365)
data = add_ml_signals(data, signal_source='random')

# Run backtest
runner = BacktestRunner(initial_cash=100000)
results = runner.run_backtest(
    data=data,
    strategy_class=MLClassificationStrategy,
    strategy_params={'ml_threshold': 0.6}
)

print(f"Total Return: {results['total_return']:.2f}%")
```

### Using Your Own Data

```python
from backtesting import prepare_data_for_backtest, quick_backtest

# Prepare data from CSV file
data = prepare_data_for_backtest(
    data_file='your_data.csv',
    signal_source='inference',  # Use ML inference
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Or use the quick backtest function
results = quick_backtest(
    data_file='your_data.csv',
    strategy_params={'ml_threshold': 0.65}
)
```

## Strategy Types

### 1. ML Classification Strategy
Trades based on machine learning classification signals.

```python
from backtesting import MLClassificationStrategy

strategy_params = {
    'ml_threshold': 0.6,        # Buy threshold
    'stop_loss': 0.05,          # 5% stop loss
    'take_profit': 0.15,        # 15% take profit
    'min_confidence': 0.7,      # Minimum confidence
    'signal_generator_type': 'ml'  # Signal source
}
```

### 2. Multi-Signal Strategy
Combines multiple signal sources with custom weights.

```python
from backtesting import MultiSignalStrategy

strategy_params = {
    'signal_weights': {
        'ml': 0.7,              # 70% ML signals
        'technical': 0.3        # 30% technical indicators
    },
    'ensemble_threshold': 0.6
}
```

### 3. Custom Strategy
Create your own strategy by inheriting from `BaseStrategy`.

```python
from backtesting import BaseStrategy
import backtrader as bt

class MyCustomStrategy(BaseStrategy):
    def _initialize_strategy(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=20)
    
    def _get_signal(self):
        if self.dataclose[0] > self.sma[0]:
            return 0.8, 0.7  # Strong buy signal
        else:
            return 0.2, 0.7  # Strong sell signal
```

## Signal Generators

### Available Signal Types

1. **ML Classification** (`'ml'`): Uses trained ML models
2. **Technical Indicators** (`'technical'`): RSI, MACD, SMA crossovers, etc.
3. **Random** (`'random'`): Random signals for testing
4. **Pre-computed** (`'precomputed'`): Use signals from data files

```python
from backtesting.signal_generators import create_signal_generator

# Create different signal generators
ml_generator = create_signal_generator('ml', model_path='path/to/model.pkl')
tech_generator = create_signal_generator('technical')
random_generator = create_signal_generator('random', seed=42)
```

## Advanced Features

### Parameter Optimization

```python
runner = BacktestRunner()

param_grid = {
    'ml_threshold': [0.55, 0.6, 0.65, 0.7],
    'stop_loss': [0.05, 0.08, 0.10],
    'take_profit': [0.15, 0.20, 0.25]
}

optimization_results = runner.run_parameter_optimization(
    data=data,
    strategy_class=MLClassificationStrategy,
    param_grid=param_grid,
    optimization_metric='total_return'
)

print(f"Best params: {optimization_results['best_params']}")
```

### Strategy Comparison

```python
strategy_configs = [
    {
        'name': 'Conservative',
        'strategy_class': MLClassificationStrategy,
        'params': {'ml_threshold': 0.7, 'stop_loss': 0.05}
    },
    {
        'name': 'Aggressive',
        'strategy_class': MLClassificationStrategy,
        'params': {'ml_threshold': 0.55, 'stop_loss': 0.10}
    }
]

comparison_df = runner.compare_strategies(data, strategy_configs)
print(comparison_df)
```

## Data Requirements

### OHLCV Data Format
Your CSV file should contain these columns:
- `open`, `high`, `low`, `close`, `volume`
- Date/time column: `timestamp`, `date`, or `datetime`

### ML Signals (Optional)
If using pre-computed signals, add these columns:
- `ml_signal`: Signal strength (0-1)
- `ml_confidence`: Confidence level (0-1)

## Integration with Existing Models

### Solana Trading Models
```python
# Use with your existing Solana classification models
from backtesting import MLClassificationStrategy

strategy_params = {
    'signal_generator_type': 'ml',
    'model_path': 'solana/models/classification_forward/best_model.pkl'
}

results = runner.run_backtest(
    data=data,
    strategy_class=MLClassificationStrategy,
    strategy_params=strategy_params
)
```

### Regime-Specific Models
```python
# Create custom strategy for regime-aware trading
class RegimeAwareStrategy(BaseStrategy):
    def _initialize_strategy(self):
        from solana.inference.regime_classification.regime_classifier_inference import RegimeClassifier
        from solana.inference.classification_forward.classification_inference import ClassificationInference
        
        self.regime_classifier = RegimeClassifier()
        self.classification_models = {
            0: ClassificationInference('models/regime_0/'),
            1: ClassificationInference('models/regime_1/'),
            2: ClassificationInference('models/regime_2/')
        }
    
    def _get_signal(self):
        # Get current regime
        regime = self.regime_classifier.predict_regime(current_features)
        
        # Use regime-specific model
        model = self.classification_models[regime]
        signal, confidence = model.predict(current_features)
        
        return signal, confidence
```

## Performance Tips

1. **Use vectorized signal generation** when possible
2. **Limit parameter optimization combinations** for faster testing
3. **Disable verbose logging** for batch operations
4. **Cache preprocessed data** for repeated backtests
5. **Use sampling** for initial parameter exploration

## Examples

Run the examples to see the framework in action:

```bash
# Simple example
python backtesting/examples/simple_example.py

# Comprehensive demo with all features
python backtesting/examples/demo_usage.py
```

## Extending the Framework

### Add New Strategy Types
1. Inherit from `BaseStrategy`
2. Implement `_initialize_strategy()` and `_get_signal()`
3. Optionally override `_should_buy()` and `_should_sell()`

### Add New Signal Generators
1. Inherit from `SignalGenerator`
2. Implement `initialize()` and `generate_signal()`
3. Add to the factory function in `signal_generators.py`

### Add New Analyzers
1. Use any Backtrader analyzer
2. Add to the analyzer map in `backtest_runner.py`
3. Implement custom result extraction logic

## Dependencies

- `backtrader`: Core backtesting engine
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: ML model support (optional)
- Your existing ML inference modules

## Migration from Original Script

The original monolithic script has been broken down as follows:

- **Strategy logic** → `base_strategy.py` + `ml_strategy.py`
- **Signal generation** → `signal_generators.py`
- **Data preparation** → `data_utils.py`
- **Backtest execution** → `backtest_runner.py`
- **Demo functionality** → `examples/`

This modular approach makes it easy to:
- **Reuse components** across different strategies
- **Test new signal sources** without changing strategy code
- **Compare strategies** systematically
- **Extend functionality** without breaking existing code
- **Maintain and debug** individual components
