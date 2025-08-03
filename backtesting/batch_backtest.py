#!/usr/bin/env python3
"""
Batch Backtest Script - Run strategy on all OHLVC signal files
"""

import backtrader as bt
import pandas as pd
import glob
import sys
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plots
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# Add strategies directory to path
sys.path.append(str(Path(__file__).parent / 'strategies'))
from simple_exit_strategy import SimpleExitStrategy

# Custom data feed
class OHLVCSignalsData(bt.feeds.PandasData):
    lines = ('safe_long_signal', 'regime_1_contrarian_signal')
    params = dict(
        datetime=None,
        open='open',
        high='high', 
        low='low',
        close='close',
        volume='volume',
        openinterest='openinterest',
        safe_long_signal='safe_long_signal',
        regime_1_contrarian_signal='regime_1_contrarian_signal',
    )

def run_single_backtest(csv_path, output_dir, strategy_params=None):
    """Run backtest on single OHLVC file"""
    
    coin_id = Path(csv_path).stem.replace('_ohlvc_signals', '')
    print(f"🪙 Processing: {coin_id}")
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        if len(df) < 10:  # Skip coins with too little data
            print(f"⚠️ Skipping {coin_id}: insufficient data ({len(df)} bars)")
            return None
        
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        
        # Add data
        data = OHLVCSignalsData(dataname=df)
        cerebro.adddata(data)
        
        # Add strategy with parameters
        if strategy_params:
            cerebro.addstrategy(SimpleExitStrategy, **strategy_params)
        else:
            cerebro.addstrategy(SimpleExitStrategy)
        
        # Set broker settings
        cerebro.broker.setcash(100)
        cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers (Returns analyzer causes math range error with volatile data)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')  # CAUSES MATH RANGE ERROR
        
        # Run backtest (disable logging)
        original_log = SimpleExitStrategy.log
        SimpleExitStrategy.log = lambda self, txt, dt=None: None  # Disable logging
        
        initial_value = cerebro.broker.getvalue()
        try:
            results = cerebro.run()[0]
            final_value = cerebro.broker.getvalue()
        except Exception as e:
            # Restore logging before re-raising
            SimpleExitStrategy.log = original_log
            raise e
        
        # Restore logging
        SimpleExitStrategy.log = original_log
        
        # Save detailed results
        results_file = output_dir / f"{coin_id}_results.csv"
        results.save_results(str(results_file))
        
        # Save plot (optional - disable if causing issues)
        plot_file = output_dir / f"{coin_id}_chart.png"
        try:
            # Skip plotting for now to avoid popup issues
            # cerebro.plot(style='candlestick', volume=False, 
            #             savefig=dict(fname=str(plot_file), dpi=150, bbox_inches='tight'))
            plot_file = None  # Disable plotting for batch processing
        except Exception as e:
            print(f"⚠️ Plot failed for {coin_id}: {e}")
            plot_file = None
        
        # Extract performance metrics
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Trade analysis
        trades = results.analyzers.trades.get_analysis()
        trade_stats = {}
        if 'total' in trades and trades['total']['total'] > 0:
            trade_stats = {
                'total_trades': trades['total']['total'],
                'winning_trades': trades['won']['total'],
                'losing_trades': trades['lost']['total'],
                'win_rate': trades['won']['total'] / trades['total']['total'] * 100,
                'avg_win': trades['won']['pnl']['average'] if 'pnl' in trades['won'] else 0,
                'avg_loss': trades['lost']['pnl']['average'] if 'pnl' in trades['lost'] else 0,
            }
        
        # Other metrics (testing DrawDown)
        sharpe = results.analyzers.sharpe.get_analysis()
        drawdown = results.analyzers.drawdown.get_analysis()
        
        return {
            'coin_id': coin_id,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe.get('sharperatio', None),
            'max_drawdown': drawdown['max']['drawdown'] if 'max' in drawdown else None,
            'data_points': len(df),
            'results_file': str(results_file),
            'plot_file': str(plot_file),
            **trade_stats
        }
        
    except Exception as e:
        print(f"❌ Error processing {coin_id}: {e}")
        return None

def main():
    """Main batch processing function"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup directories
    input_dir = Path('backtesting/data/ohlvc_signals')
    output_dir = Path(f'backtesting/results/batch_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting batch backtest")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Find all OHLVC signal files
    signal_files = list(input_dir.glob('*_ohlvc_signals.csv'))
    print(f"📊 Found {len(signal_files)} signal files")
    
    if len(signal_files) == 0:
        print("❌ No signal files found!")
        return
    
    # Strategy parameters
    strategy_params = {
        'hold_bars': 2,
        'contrarian_size_sol': 1,
        'safe_long_cash_pct': 0.9
    }
    
    print(f"⚙️ Strategy parameters: {strategy_params}")
    
    # Process all files
    results = []
    successful = 0
    failed = 0
    failed_coins = []
    
    for i, csv_file in enumerate(signal_files, 1):
        print(f"\n[{i}/{len(signal_files)}] ", end="")
        
        result = run_single_backtest(csv_file, output_dir, strategy_params)
        
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
            coin_id = Path(csv_file).stem.replace('_ohlvc_signals', '')
            failed_coins.append(coin_id)
    
    # Save summary results
    if results:
        summary_df = pd.DataFrame(results)
        summary_file = output_dir / 'batch_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n📊 BATCH SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(signal_files)}")
        print(f"Successful backtests: {successful}")
        print(f"Failed backtests: {failed}")
        if failed_coins:
            print(f"Failed coins: {', '.join(failed_coins)}")
        print(f"Success rate: {successful/len(signal_files)*100:.1f}%")
        
        if len(summary_df) > 0:
            print(f"\n📈 PERFORMANCE SUMMARY")
            print(f"Average return: {summary_df['total_return'].mean():.2f}%")
            print(f"Median return: {summary_df['total_return'].median():.2f}%")
            print(f"Best performer: {summary_df.loc[summary_df['total_return'].idxmax(), 'coin_id']} ({summary_df['total_return'].max():.2f}%)")
            print(f"Worst performer: {summary_df.loc[summary_df['total_return'].idxmin(), 'coin_id']} ({summary_df['total_return'].min():.2f}%)")
            
            # Filter for coins with trades
            traded_coins = summary_df[summary_df['total_trades'] > 0]
            if len(traded_coins) > 0:
                print(f"Coins with trades: {len(traded_coins)}/{len(summary_df)}")
                print(f"Average win rate: {traded_coins['win_rate'].mean():.1f}%")
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"📋 Summary file: {summary_file}")
    
    else:
        print("❌ No successful backtests!")

if __name__ == "__main__":
    main()