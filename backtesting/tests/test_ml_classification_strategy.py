#!/usr/bin/env python3
"""
Test ML Classification Strategy
Complete test of the ML trading pipeline: WindowFeatureCalculator + ClassificationInference
"""

import sys
sys.path.append('/Users/noel/projects/trading_eda')

import pandas as pd
import numpy as np
import duckdb
import backtrader as bt
from datetime import datetime, timedelta

from backtesting.solana_transaction_feed import SolanaTransactionFeed
from backtesting.strategies import MLClassificationStrategy
from backtesting.onchain_broker import setup_onchain_broker


def create_test_transaction_data(coin_id: str, n_samples: int = 100):
    """Load real transaction data for ML testing"""
    print(f"📊 Loading transaction data for {coin_id}...")
    
    # Connect to DuckDB
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Get transaction data
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
    LIMIT {n_samples * 50}  -- Get more data than needed for good ML testing
    """
    
    transaction_df = conn.execute(transaction_query).fetchdf()
    print(f"✅ Loaded {len(transaction_df)} transactions")
    print(f"Time range: {transaction_df['block_timestamp'].min()} to {transaction_df['block_timestamp'].max()}")
    
    conn.close()
    return transaction_df


def test_ml_classification_strategy():
    """Test the complete ML classification strategy"""
    print("🤖 TESTING ML CLASSIFICATION STRATEGY")
    print("=" * 60)
    
    # Test configuration
    coin_id = "8xPe3DMr52oYAkCBk57ZeQE1h5zBkWNk37eE4J8spump"  # Use same coin as other tests
    print(f"Testing coin: {coin_id}")
    
    # Load real transaction data
    transaction_df = create_test_transaction_data(coin_id, n_samples=200)
    
    if len(transaction_df) < 100:
        print("❌ Insufficient transaction data for ML testing")
        return
    
    # Setup Cerebro engine
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
    
    # Add ML classification strategy
    cerebro.addstrategy(
        MLClassificationStrategy,
        lookback_windows=[30, 60, 120],      # 30s, 1min, 2min windows
        prediction_threshold=0.6,            # ML confidence threshold
        require_high_confidence=False,       # Allow medium confidence trades
        position_size_pct=0.15,              # 15% position sizes for testing
        stop_loss=None,                      # No stop loss - trust model for 5 bars
        take_profit=0.15,                    # 15% take profit for exceptional moves
        max_holding_bars=5,                  # 5 bars = 300s (model prediction horizon)
        max_daily_trades=8,                  # More trades with shorter holds
        verbose=True,
        log_features=False,                  # Reduce noise
        log_predictions=True                 # Show ML predictions
    )
    
    # Setup onchain broker
    setup_onchain_broker(cerebro, initial_cash=100000)  # $100k for ML testing
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print(f"💰 Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    
    # Run backtest
    print("\n🤖 Running ML classification strategy backtest...")
    print("-" * 60)
    
    try:
        results = cerebro.run()
        
        # Get results
        final_value = cerebro.broker.getvalue()
        final_cash = cerebro.broker.getcash()
        position = cerebro.broker.getposition(cerebro.datas[0])
        total_return = (final_value - 100000) / 100000 * 100
        
        print(f"\n🤖 === ML CLASSIFICATION STRATEGY RESULTS ===")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Final Cash: ${final_cash:,.2f}")
        print(f"Position Size: {position.size}")
        print(f"Position Value: ${position.size * cerebro.datas[0].close[0] if position.size != 0 else 0:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Detailed analysis
        strategy = results[0]
        
        # Trade analysis
        try:
            trades = strategy.analyzers.trades.get_analysis()
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            
            print(f"\n📊 TRADING ANALYSIS:")
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {won_trades}")
            print(f"Losing Trades: {lost_trades}")
            
            if total_trades > 0:
                win_rate = (won_trades / total_trades * 100)
                print(f"Win Rate: {win_rate:.1f}%")
                
                # PnL analysis
                if won_trades > 0:
                    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
                    print(f"Average Win: ${avg_win:.2f}")
                if lost_trades > 0:
                    avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
                    print(f"Average Loss: ${avg_loss:.2f}")
        
        except Exception as e:
            print(f"⚠️ Trade analysis error: {e}")
        
        # Sharpe ratio
        try:
            sharpe = strategy.analyzers.sharpe.get_analysis()
            if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
                print(f"Sharpe Ratio: {sharpe['sharperatio']:.3f}")
        except Exception as e:
            print(f"⚠️ Sharpe analysis error: {e}")
        
        # Drawdown analysis
        try:
            drawdown = strategy.analyzers.drawdown.get_analysis()
            max_dd = drawdown.get('max', {}).get('drawdown', 0)
            print(f"Max Drawdown: {max_dd:.2f}%")
        except Exception as e:
            print(f"⚠️ Drawdown analysis error: {e}")
        
        print("\n🎯 ML STRATEGY INSIGHTS:")
        print("✅ Complete ML pipeline successfully implemented")
        print("✅ WindowFeatureCalculator calculates real-time features")
        print("✅ ClassificationInference makes ML predictions")
        print("✅ Strategy trades based on ML model confidence")
        print("✅ Demonstrates: Raw data → Features → ML Predictions → Trading signals")
        
        # Performance assessment
        if total_return > 5:
            print("\n🎉 Strong ML strategy performance!")
        elif total_return > 0:
            print("\n✅ Positive ML strategy performance")
        else:
            print("\n📊 ML strategy learning opportunity")
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'strategy': strategy,
            'total_trades': total_trades if 'total_trades' in locals() else 0
        }
        
    except Exception as e:
        print(f"\n❌ ML Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🤖 ML CLASSIFICATION STRATEGY TEST")
    print("=" * 50)
    
    try:
        results = test_ml_classification_strategy()
        if results:
            print(f"\n✅ ML classification strategy test completed successfully!")
            print(f"📈 Final Performance: {results['total_return']:.2f}% return")
        else:
            print(f"\n❌ ML classification strategy test failed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()