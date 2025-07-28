#!/usr/bin/env python3
"""
ML Classification Strategy Visualization Test
Comprehensive backtesting with visualizations for strategy analysis
"""

import sys
sys.path.append('/Users/noel/projects/trading_eda')

import pandas as pd
import numpy as np
import duckdb
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from backtesting.solana_transaction_feed import SolanaTransactionFeed
from backtesting.strategies import MLClassificationStrategy
from backtesting.onchain_broker import setup_onchain_broker


class DetailedTradeAnalyzer(bt.Analyzer):
    """Enhanced trade analyzer that captures detailed trade information"""
    
    def __init__(self):
        self.trades = []
        self.trade_sequence = []
        self.portfolio_values = []
        self.ml_predictions = []
        
    def notify_trade(self, trade):
        if trade.isclosed:
            trade_data = {
                'entry_date': trade.open_datetime,
                'exit_date': trade.close_datetime,
                'entry_price': trade.open_price if hasattr(trade, 'open_price') else trade.price,
                'exit_price': trade.close_price if hasattr(trade, 'close_price') else trade.price,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'size': trade.size,
                'bars_held': trade.barclose - trade.baropen + 1,
                'return_pct': (trade.pnlcomm / abs(trade.value) * 100) if trade.value != 0 else 0
            }
            self.trades.append(trade_data)
    
    def next(self):
        # Track portfolio value progression
        try:
            # Use strategy reference to get broker
            strategy = self.strategy if hasattr(self, 'strategy') else self._parent
            if strategy and hasattr(strategy, 'broker'):
                portfolio_value = strategy.broker.getvalue()
                current_datetime = strategy.datas[0].datetime.datetime(0)
                cash_value = strategy.broker.getcash()
                
                self.portfolio_values.append({
                    'datetime': current_datetime,
                    'value': portfolio_value,
                    'cash': cash_value
                })
        except Exception as e:
            # Skip if we can't access broker info
            pass
    
    def get_analysis(self):
        return {
            'trades': self.trades,
            'portfolio_progression': self.portfolio_values
        }


def create_real_test_data():
    """Load the same real data used in the successful ML test"""
    print(f"📊 Loading real transaction data (same as successful ML test)...")
    
    # Use the exact same coin and query as the working ML test
    coin_id = "4vRuo7xrDXtQLHKUR43zMKPgJtns88zBeXhDtWjCpump"
    
    # Connect to DuckDB
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Use the exact same query as test_ml_classification_strategy.py
    transaction_query = f"""
    SELECT 
        block_timestamp,
        mint,
        swapper,
        succeeded,
        CASE WHEN mint = swap_to_mint THEN 1 ELSE 0 END as is_buy,
        CASE 
            WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_from_amount
            WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_to_amount
            ELSE 0.0
        END as sol_amount,
        swap_from_amount,
        swap_to_amount,
        swap_from_mint,
        swap_to_mint
    FROM first_day_trades
    WHERE mint = '{coin_id}'
    AND succeeded = TRUE
    AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
         OR swap_to_mint = 'So11111111111111111111111111111111111111112')
    AND mint != 'So11111111111111111111111111111111111111112'
    ORDER BY block_timestamp
    """
    
    transaction_df = conn.execute(transaction_query).fetchdf()
    print(f"✅ Loaded {len(transaction_df)} transactions for {coin_id}")
    print(f"Time range: {transaction_df['block_timestamp'].min()} to {transaction_df['block_timestamp'].max()}")
    
    conn.close()
    return transaction_df, coin_id


def run_ml_strategy_backtest(transaction_df: pd.DataFrame, coin_id: str):
    """Run ML strategy backtest with detailed tracking"""
    print(f"\n🤖 Running ML strategy backtest...")
    
    # Setup Cerebro
    cerebro = bt.Cerebro()
    
    # Add transaction data feed
    transaction_feed = SolanaTransactionFeed(
        transaction_data=transaction_df,
        aggregation_window=60,  # 1-minute windows
        datetime_col='block_timestamp',
        volume_col='sol_amount',
        direction_col='is_buy',
        trader_col='swapper',
        succeed_col='succeeded'
    )
    
    cerebro.adddata(transaction_feed)
    
    # Add ML classification strategy (exact same parameters as successful test)
    cerebro.addstrategy(
        MLClassificationStrategy,
        lookback_windows=[30, 60, 120],      # 30s, 1min, 2min windows
        prediction_threshold=0.8,            # ML confidence threshold
        require_high_confidence=False,       # Allow medium confidence trades
        position_size_pct=0.8,              # 15% position sizes for testing
        stop_loss=None,                      # No stop loss - trust model for 5 bars
        take_profit=0.15,                    # 15% take profit for exceptional moves
        max_holding_bars=5,                  # 5 bars = 300s (model prediction horizon)
        max_daily_trades=500,                  # More trades with shorter holds
        verbose=True,                        # Show strategy details
        log_features=False,                  # Reduce noise
        log_predictions=True                 # Show ML predictions
    )
    
    # Setup onchain broker
    setup_onchain_broker(cerebro, initial_cash=100000)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(DetailedTradeAnalyzer, _name='detailed')
    
    print(f"💰 Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    
    # Run backtest
    results = cerebro.run()
    
    # Get results
    final_value = cerebro.broker.getvalue()
    initial_cash = 100000
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    print(f"💰 Final Portfolio Value: ${final_value:,.2f}")
    print(f"📈 Total Return: {total_return:.2f}%")
    
    return results[0], cerebro


def create_visualizations(strategy, cerebro, coin_id: str):
    """Create comprehensive visualizations of strategy performance"""
    print(f"\n📊 Creating visualizations...")
    
    # Get analysis data
    trades_analysis = strategy.analyzers.trades.get_analysis()
    detailed_analysis = strategy.analyzers.detailed.get_analysis()
    
    # Setup the visualization
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Portfolio Value Progression
    plt.subplot(3, 4, 1)
    portfolio_data = detailed_analysis['portfolio_progression']
    if portfolio_data:
        dates = [p['datetime'] for p in portfolio_data]
        values = [p['value'] for p in portfolio_data]
        plt.plot(dates, values, 'b-', linewidth=2)
        plt.title('Portfolio Value Over Time')
        plt.ylabel('Portfolio Value ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 2. Win/Loss Distribution
    plt.subplot(3, 4, 2)
    won_trades = trades_analysis.get('won', {}).get('total', 0)
    lost_trades = trades_analysis.get('lost', {}).get('total', 0)
    if won_trades + lost_trades > 0:
        plt.pie([won_trades, lost_trades], 
                labels=[f'Won ({won_trades})', f'Lost ({lost_trades})'], 
                autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c'],
                startangle=90)
        plt.title('Win/Loss Ratio')
    
    # 3. Trade P&L Distribution
    plt.subplot(3, 4, 3)
    trades_list = detailed_analysis['trades']
    if trades_list:
        pnl_values = [t['pnlcomm'] for t in trades_list]
        winning_trades = [p for p in pnl_values if p > 0]
        losing_trades = [p for p in pnl_values if p <= 0]
        
        plt.hist([winning_trades, losing_trades], 
                bins=20, 
                label=['Wins', 'Losses'], 
                color=['#2ecc71', '#e74c3c'], 
                alpha=0.7)
        plt.title('P&L Distribution')
        plt.xlabel('P&L ($)')
        plt.ylabel('Number of Trades')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Trade Returns Percentage
    plt.subplot(3, 4, 4)
    if trades_list:
        returns = [t['return_pct'] for t in trades_list]
        plt.hist(returns, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
        plt.title('Trade Returns Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Number of Trades')
        plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.1f}%')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Trade Duration Analysis
    plt.subplot(3, 4, 5)
    if trades_list:
        durations = [t['bars_held'] for t in trades_list]
        duration_counts = pd.Series(durations).value_counts().sort_index()
        plt.bar(duration_counts.index, duration_counts.values, color='#9b59b6', alpha=0.7)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Bars Held')
        plt.ylabel('Number of Trades')
        plt.grid(True, alpha=0.3)
    
    # 6. Cumulative Returns
    plt.subplot(3, 4, 6)
    if trades_list:
        cumulative_pnl = np.cumsum([t['pnlcomm'] for t in trades_list])
        trade_numbers = range(1, len(cumulative_pnl) + 1)
        plt.plot(trade_numbers, cumulative_pnl, 'g-', linewidth=2, marker='o', markersize=4)
        plt.title('Cumulative P&L')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True, alpha=0.3)
    
    # 7. Monthly Performance (if data spans multiple days)
    plt.subplot(3, 4, 7)
    if trades_list:
        trade_dates = [t['entry_date'] for t in trades_list]
        trade_pnl = [t['pnlcomm'] for t in trades_list]
        
        # Group by day if we have multi-day data
        daily_pnl = {}
        for date, pnl in zip(trade_dates, trade_pnl):
            day = date.date() if hasattr(date, 'date') else date
            daily_pnl[day] = daily_pnl.get(day, 0) + pnl
        
        if len(daily_pnl) > 1:
            days = list(daily_pnl.keys())
            pnl_values = list(daily_pnl.values())
            plt.bar(range(len(days)), pnl_values, color='#f39c12', alpha=0.7)
            plt.title('Daily P&L')
            plt.xlabel('Day')
            plt.ylabel('P&L ($)')
            plt.xticks(range(len(days)), [str(d) for d in days], rotation=45)
        else:
            plt.text(0.5, 0.5, 'Single Day Data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Daily P&L (Single Day)')
        plt.grid(True, alpha=0.3)
    
    # 8. Risk Metrics Summary
    plt.subplot(3, 4, 8)
    plt.axis('off')
    
    # Calculate key metrics
    total_trades = trades_analysis.get('total', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    avg_win = trades_analysis.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = trades_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
    profit_factor = abs(avg_win * won_trades / (avg_loss * lost_trades)) if lost_trades > 0 and avg_loss != 0 else float('inf')
    
    # Drawdown
    try:
        drawdown = strategy.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
    except:
        max_dd = 0
    
    # Sharpe ratio
    try:
        sharpe = strategy.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', 0) if sharpe.get('sharperatio') is not None else 0
    except:
        sharpe_ratio = 0
    
    metrics_text = f"""
STRATEGY PERFORMANCE METRICS

Total Trades: {total_trades}
Win Rate: {win_rate:.1f}%
Profit Factor: {profit_factor:.2f}

Average Win: ${avg_win:,.2f}
Average Loss: ${avg_loss:,.2f}
Max Drawdown: {max_dd:.2f}%

Sharpe Ratio: {sharpe_ratio:.3f}
Final Return: {((cerebro.broker.getvalue() - 100000) / 100000 * 100):.2f}%

Coin: {coin_id[:8]}...
Window: 5 bars (300s)
Model Threshold: 0.6
"""
    
    plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 9. Entry vs Exit Price Scatter
    plt.subplot(3, 4, 9)
    if trades_list:
        entry_prices = [t['entry_price'] for t in trades_list]
        exit_prices = [t['exit_price'] for t in trades_list]
        colors = ['green' if t['pnlcomm'] > 0 else 'red' for t in trades_list]
        
        plt.scatter(entry_prices, exit_prices, c=colors, alpha=0.6, s=50)
        plt.plot([min(entry_prices), max(entry_prices)], [min(entry_prices), max(entry_prices)], 'k--', alpha=0.5)
        plt.title('Entry vs Exit Prices')
        plt.xlabel('Entry Price')
        plt.ylabel('Exit Price')
        plt.grid(True, alpha=0.3)
    
    # 10. Trade Size Analysis
    plt.subplot(3, 4, 10)
    if trades_list:
        trade_sizes = [abs(t['size']) for t in trades_list]
        trade_returns = [t['return_pct'] for t in trades_list]
        
        plt.scatter(trade_sizes, trade_returns, alpha=0.6, s=50, c='purple')
        plt.title('Trade Size vs Returns')
        plt.xlabel('Position Size')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
    
    # 11. Weekly Performance Heatmap (if applicable)
    plt.subplot(3, 4, 11)
    if trades_list and len(trades_list) > 5:
        # Create hourly heatmap
        hour_performance = {}
        for trade in trades_list:
            hour = trade['entry_date'].hour if hasattr(trade['entry_date'], 'hour') else 12
            hour_performance[hour] = hour_performance.get(hour, []) + [trade['return_pct']]
        
        hours = sorted(hour_performance.keys())
        avg_returns = [np.mean(hour_performance[h]) for h in hours]
        
        plt.bar(hours, avg_returns, color='orange', alpha=0.7)
        plt.title('Performance by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Avg Return (%)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient Data\nfor Hourly Analysis', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Performance by Hour')
    
    # 12. Running Win Rate
    plt.subplot(3, 4, 12)
    if trades_list:
        running_wins = []
        running_total = []
        running_win_rate = []
        
        wins = 0
        total = 0
        
        for trade in trades_list:
            total += 1
            if trade['pnlcomm'] > 0:
                wins += 1
            
            running_total.append(total)
            running_wins.append(wins)
            running_win_rate.append(wins / total * 100)
        
        plt.plot(running_total, running_win_rate, 'b-', linewidth=2)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Line')
        plt.title('Running Win Rate')
        plt.xlabel('Trade Number')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'ML Classification Strategy Analysis - {coin_id[:8]}...', fontsize=16, y=0.98)
    
    # Save the plot
    plt.savefig('/Users/noel/projects/trading_eda/ml_strategy_analysis.png', dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: ml_strategy_analysis.png")
    
    plt.show()


def create_backtrader_chart(cerebro, coin_id: str):
    """Create Backtrader's built-in candlestick chart with trade markers"""
    print(f"\n📈 Creating Backtrader candlestick chart...")
    
    try:
        # Create the chart
        cerebro.plot(
            style='candlestick',           # Candlestick chart
            volume=True,                   # Show volume subplot
            figsize=(16, 10),             # Large figure size
            plotdist=0.1,                 # Distance between subplots
            barup='#2ecc71',              # Green for up candles
            bardown='#e74c3c',            # Red for down candles
            volup='#27ae60',              # Dark green for volume up
            voldown='#c0392b',            # Dark red for volume down
            iplot=False,                  # Don't show inline (save instead)
        )
        
        print(f"📈 Backtrader chart created successfully!")
        print(f"   - Candlestick price chart with trade markers")
        print(f"   - Volume subplot")
        print(f"   - Buy/sell signals marked")
        print(f"   - Coin: {coin_id[:8]}...")
        
    except Exception as e:
        print(f"⚠️ Could not create Backtrader chart: {e}")
        print("   This might be due to matplotlib backend issues")


def test_ml_visualization():
    """Main test function for ML strategy visualization"""
    print("🎯 ML CLASSIFICATION STRATEGY VISUALIZATION TEST")
    print("=" * 70)
    
    try:
        # Load real dataset (same as successful ML test)
        transaction_df, coin_id = create_real_test_data()
        
        if len(transaction_df) < 1000:
            print("⚠️ Dataset too small for meaningful visualization")
            return
        
        # Run backtest
        strategy, cerebro = run_ml_strategy_backtest(transaction_df, coin_id)
        
        # Create custom visualizations
        create_visualizations(strategy, cerebro, coin_id)
        
        # Create Backtrader's built-in candlestick chart
        create_backtrader_chart(cerebro, coin_id)
        
        print(f"\n✅ Visualization test completed successfully!")
        print(f"📈 Check ml_strategy_analysis.png for detailed charts")
        print(f"📊 Backtrader candlestick chart with trade markers generated")
        
    except Exception as e:
        print(f"\n❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ml_visualization()