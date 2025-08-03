import backtrader as bt
import pandas as pd
import sys
from pathlib import Path

# Add strategies directory to path
sys.path.append(str(Path(__file__).parent.parent / 'strategies'))

from simple_exit_strategy import SimpleExitStrategy

# Load the OHLVC signals data
# 8PZC9VjQadfmpaXrF1neiHT9ZmqXUABZF7nzE2ai4mGa, GcmsHHG41giJYACYKrdxZp3kdKMZfg1UB7LaYzCzPxLJ
csv_path = 'backtesting/data/ohlvc_signals/GcmsHHG41giJYACYKrdxZp3kdKMZfg1UB7LaYzCzPxLJ_ohlvc_signals.csv'
df = pd.read_csv(csv_path)

# Convert datetime and set as index
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Custom data feed with safe_long_signal
class OHLVCSignalsData(bt.feeds.PandasData):
    lines = ('safe_long_signal','regime_1_contrarian_signal',)
    params = dict(
        datetime=None,  # use index
        open='open',
        high='high', 
        low='low',
        close='close',
        volume='volume',
        openinterest='openinterest',
        safe_long_signal='safe_long_signal',
        regime_1_contrarian_signal='regime_1_contrarian_signal',
    )

# Strategy class is now imported from simple_exit_strategy.py

# Run backtest
cerebro = bt.Cerebro()
data = OHLVCSignalsData(dataname=df)
cerebro.adddata(data)
cerebro.addstrategy(SimpleExitStrategy, hold_bars=2)
cerebro.broker.setcash(100)
cerebro.broker.setcommission(commission=0.001)

# analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='dd')
cerebro.addanalyzer(bt.analyzers.TimeReturn,  _name='rets')
print(f"Starting Value: {cerebro.broker.getvalue():.2f}")
results = cerebro.run()[0]
print(f"Final Value: {cerebro.broker.getvalue():.2f}")

# Save detailed results
results.save_results('test_backtest_results.csv')

# print headline stats
sharpe_analysis = results.analyzers.sharpe.get_analysis()
dd_analysis = results.analyzers.dd.get_analysis()
rets_analysis = results.analyzers.rets.get_analysis()

print('Sharpe  (daily):', sharpe_analysis.get('sharperatio', 'N/A'))
print('Max DD  (%):    ', dd_analysis['max']['drawdown'])
print('TimeReturn keys:', list(rets_analysis.keys()))  # Debug: see available keys

# pretty picture
cerebro.plot(style='candlestick', volume=True)