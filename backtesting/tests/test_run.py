import backtrader as bt
import pandas as pd

# Load the OHLVC signals data
csv_path = 'backtesting/data/ohlvc_signals/2Kk16bkuFH8dsd117feYaqPjBYrF8NC5GCM2VMyKpump_ohlvc_signals.csv'
df = pd.read_csv(csv_path)

# Convert datetime and set as index
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Custom data feed with safe_long_signal
class OHLVCSignalsData(bt.feeds.PandasData):
    lines = ('safe_long_signal',)
    params = dict(
        datetime=None,  # use index
        open='open',
        high='high', 
        low='low',
        close='close',
        volume='volume',
        openinterest='openinterest',
        safe_long_signal='safe_long_signal'
    )

# Simple strategy: buy on safe_long_signal
class SimpleSignalStrategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.strftime('%Y-%m-%d %H:%M:%S'), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.safe_long_signal = self.datas[0].safe_long_signal

    def next(self):
        self.log('Close, %.2f, Signal, %d, Portfolio, %.2f' % (
            self.dataclose[0], 
            self.safe_long_signal[0], 
            self.broker.getvalue()
        ))
        
        if self.safe_long_signal[0] == 1 and not self.position:
            self.buy()
            self.log('BUY CREATE, %.2f' % self.dataclose[0])
        elif self.position and len(self.data) > 10:
            self.close()
            self.log('SELL CREATE, %.2f' % self.dataclose[0])

# Run backtest
cerebro = bt.Cerebro()
data = OHLVCSignalsData(dataname=df)
cerebro.adddata(data)
cerebro.addstrategy(SimpleSignalStrategy)
cerebro.broker.setcash(10000)

# analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown,    _name='dd')
cerebro.addanalyzer(bt.analyzers.TimeReturn,  _name='rets')
print(f"Starting Value: {cerebro.broker.getvalue():.2f}")
results = cerebro.run()[0]
print(f"Final Value: {cerebro.broker.getvalue():.2f}")

# print headline stats
sharpe_analysis = results.analyzers.sharpe.get_analysis()
dd_analysis = results.analyzers.dd.get_analysis()
rets_analysis = results.analyzers.rets.get_analysis()

print('Sharpe  (daily):', sharpe_analysis.get('sharperatio', 'N/A'))
print('Max DD  (%):    ', dd_analysis['max']['drawdown'])
print('TimeReturn keys:', list(rets_analysis.keys()))  # Debug: see available keys

# pretty picture
cerebro.plot(style='candlestick', volume=True)