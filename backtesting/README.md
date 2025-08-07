# Backtesting Framework with Execution Validation

A complete backtesting system for ML-based trading strategies using validated OHLVC data with realistic execution constraints.

## Current Status (from progress.md)

### ✅ **Completed Components**

**Models & Inference:**
- Classification models with AUC 0.66 (forward 300s prediction, lookback 120s)
- Regime-based classification models from clustering
- Reusable feature engineering and inference pipelines

**Data Pipeline:**
- OHLVC signal conversion from per-transaction data
- **Execution validation system** with buy/sell feasibility checks
- 3-bar holding period with volume and price validation
- 827 validated CSV files with realistic trading constraints

**Backtesting Framework:**  
- Backtrader-based implementation with custom data feeds
- Simple exit strategy using execution validation signals
- Batch processing for all validated coins
- **Sanity checks** for open positions and portfolio value rollback

### 📊 **Latest Results**
- **Second Round** (3-bar hold + execution validation):
  - Average return: **45.62%**
  - Median return: **2.01%**
  - Best: 4qFgyTrhaMtqST4qxiy1uEmPsQeGtpFb1uyUtWN4pump (**4179.10%**)
  - Coins with trades: **742/815**
  - Win rate: **59.4%**

## Architecture

```
backtesting/
├── data/
│   ├── convert_to_ohlvc_signals.py     # ML signals → OHLVC conversion
│   ├── add_execution_validation.py     # Execution feasibility validation  
│   ├── ohlvc_signals/                  # Original OHLVC + ML signals
│   └── ohlvc_signals_validated/        # With execution validation
├── strategies/
│   └── simple_exit_strategy.py         # Uses can_execute_buy/sell signals
├── batch_backtest.py                   # Batch processing with sanity checks
└── results/                            # Backtest outputs and analysis
```

## Execution Validation

### **Problems**
To avoid unrealistic executions and save time on implementing with backtrader framework for situations:
- No market volume exists at signal time
- Price moves unfavorably before execution
- Positions remain open at data end

### **Validation System**
Each OHLVC signal file gets three additional columns:

1. **`can_execute_buy`** (binary): Pre-validates buy feasibility
   - Next candle has sufficient volume (≥ fixed SOL amount)
   - Next candle's low < open (favorable price movement)

2. **`can_execute_sell`** (binary): Pre-validates sell feasibility  
   - Finds next candle with adequate volume for position exit
   - Respects 3-bar minimum holding period
   - Uses lookahead up to 10 bars to find liquidity

3. **`coin_size`** (float): Tracks position size throughout trade lifecycle
   - Calculated as `fixed_sol_amount / execution_price`
   - Maintained during holding period
   - Reset to 0 after sell execution

### **Sanity Checks**
Batch backtesting includes position validation:
- **Open Position Detection**: Identifies unclosed trades at data end
- **Portfolio Rollback**: Reverts final value to last successful buy
- **Clean Reporting**: Separates realistic vs phantom returns

## Usage

### **Generate Validated Data**
```bash
# Process all OHLVC signal files with execution validation
python backtesting/data/add_execution_validation.py --holding-period 3
```

### **Run Batch Backtesting**  
```bash
# Backtest all validated files with sanity checks
python backtesting/batch_backtest.py
```

### **Analyze Results**
```bash
# Plot performance vs data points
python backtesting/results/plot_final_value_vs_data_points.py
```

## Data Flow

1. **Raw Signals**: `ohlvc_signals/*.csv` (OHLVC + ML predictions)
2. **Validation**: Add execution feasibility columns
3. **Strategy**: Use `can_execute_buy/sell` instead of raw ML signals  
4. **Backtesting**: Run with portfolio sanity checks
5. **Analysis**: Clean performance metrics excluding phantom trades

## Strategy Logic

**SimpleExitStrategy** now uses validated signals:

```python
# Entry: Only when can_execute_buy = 1
if self.can_execute_buy[0] == 1:
    # Uses original position sizing logic
    size = self.params.safe_long_size / self.data.close[0]
    self.order_open = self.buy(size=size)

# Exit: Only when can_execute_sell = 1  
if self.can_execute_sell[0] == 1:
    self.order_close = self.close()
```

**ML signals kept for logging/analysis only.**

## Configuration

**Default Parameters:**
- Fixed SOL amount: **10.0** per trade
- Holding period: **3 bars** minimum
- Lookahead window: **10 bars** for sell validation
- Commission: **0.1%**

**Strategy Parameters:**
- `contrarian_size_sol`: 1 SOL for regime contrarian trades
- `safe_long_size`: 10 SOL for safe long trades

## Results Analysis

**Performance Improvement with Validation:**
- **Before**: 102.78% avg return (likely inflated by phantom trades)
- **After**: 45.62% avg return (realistic execution constraints)
- **Insight**: ~40% of previous returns were from unrealistic executions

**Trade Statistics:**
- **Execution rate**: 742/815 coins had executable trades (91%)
- **Win rate**: 59.4% of executed trades profitable
- **Best performer**: 4179% return (still exceptional but validated)

## Next Steps (from TODO)

1. **Model Improvement**: Add volatility and price action features
2. **New Data**: Pull fresh coin data from Flipside for testing  
3. **Strategy Enhancement**: Develop more sophisticated trading logic
4. **Parameter Optimization**: Fine-tune holding periods and thresholds

## Dependencies

- `backtrader`: Core backtesting engine
- `pandas`: Data manipulation  
- `numpy`: Numerical operations
- `matplotlib`: Results visualization
- Custom ML inference modules (classification_forward, regime_clustering)

The framework ensures **realistic backtesting results** by validating every trade's feasibility before execution, preventing the phantom profits common in traditional backtesting systems.