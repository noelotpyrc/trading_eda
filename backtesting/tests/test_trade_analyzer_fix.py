#!/usr/bin/env python3
"""
Test TradeAnalyzer with OnchainBroker
Verify that our broker correctly notifies TradeAnalyzer about completed trades
"""

import sys
sys.path.append('/Users/noel/projects/trading_eda')

import pandas as pd
import backtrader as bt
from datetime import datetime, timedelta

from backtesting.solana_transaction_feed import SolanaTransactionFeed
from backtesting.onchain_broker import setup_onchain_broker


class SimpleTestStrategy(bt.Strategy):
    """Simple strategy to test TradeAnalyzer integration"""
    
    def __init__(self):
        self.order = None
        self.bar_count = 0
        print("🤖 SimpleTestStrategy initialized")
    
    def next(self):
        self.bar_count += 1
        
        # Simple trading logic: buy at bar 2, sell at bar 5
        if self.bar_count == 2 and not self.position:
            print(f"📊 Bar {self.bar_count}: Placing BUY order")
            self.order = self.buy(size=1e-7)  # Very small size for scaled prices
        
        elif self.bar_count == 5 and self.position:
            print(f"📊 Bar {self.bar_count}: Placing SELL order")
            self.order = self.sell(size=self.position.size)
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"✅ BUY EXECUTED: {order.executed.size:.6f} shares at {order.executed.price:.2f}")
            else:
                print(f"✅ SELL EXECUTED: {order.executed.size:.6f} shares at {order.executed.price:.2f}")
    
    def notify_trade(self, trade):
        if trade.isclosed:
            pnl_pct = (trade.pnlcomm/trade.value*100) if trade.value != 0 else 0.0
            print(f"🎯 TRADE COMPLETED: PnL={trade.pnl:.2f}, PnL%={pnl_pct:.2f}%")
            print(f"   Entry: {trade.open_datetime} at {trade.price:.2f}")
            print(f"   Exit: {trade.close_datetime} at {trade.price:.2f}")


def test_trade_analyzer():
    """Test TradeAnalyzer with OnchainBroker"""
    print("🔍 TESTING TRADEANALYZER WITH ONCHAIN BROKER")
    print("=" * 60)
    
    # Create simple test data - more transactions to get more bars
    timestamps = [datetime(2024, 1, 1, 10, i, j*10) for i in range(20) for j in range(3)]
    
    test_data = []
    for i, ts in enumerate(timestamps):
        test_data.append({
            'block_timestamp': ts,
            'swap_from_mint': 'So11111111111111111111111111111111111111112',
            'swap_to_mint': 'test_token',
            'swap_from_amount': 10.0 + i,
            'swap_to_amount': 0.1,  # Fixed token amount
            'succeeded': True,
            'is_buy': True,
            'sol_amount': 10.0 + i,
            'swapper': f'wallet_{i}'
        })
    
    transaction_df = pd.DataFrame(test_data)
    print(f"📊 Created {len(transaction_df)} test transactions")
    
    # Setup Cerebro
    cerebro = bt.Cerebro()
    
    # Add data feed
    data_feed = SolanaTransactionFeed(
        transaction_data=transaction_df,
        aggregation_window=60,
        datetime_col='block_timestamp',
        volume_col='sol_amount',
        direction_col='is_buy',
        trader_col='swapper',
        succeed_col='succeeded'
    )
    cerebro.adddata(data_feed)
    
    # Add strategy
    cerebro.addstrategy(SimpleTestStrategy)
    
    # Setup OnchainBroker
    setup_onchain_broker(cerebro, initial_cash=100000)
    
    # Enable debug mode for trade tracking
    cerebro.broker._debug_trades = True
    
    # Add TradeAnalyzer
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"💰 Starting Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    
    # Run backtest
    print("\\n🔄 Running backtest...")
    results = cerebro.run()
    
    final_value = cerebro.broker.getvalue()
    print(f"💰 Final Portfolio Value: ${final_value:,.2f}")
    
    # Analyze results
    strategy = results[0]
    trade_analysis = strategy.analyzers.trades.get_analysis()
    
    print(f"\\n📊 TRADE ANALYZER RESULTS:")
    print(f"Total Trades: {trade_analysis.get('total', {}).get('total', 0)}")
    print(f"Winning Trades: {trade_analysis.get('won', {}).get('total', 0)}")
    print(f"Losing Trades: {trade_analysis.get('lost', {}).get('total', 0)}")
    
    if 'total' in trade_analysis and trade_analysis['total']['total'] > 0:
        print("✅ TradeAnalyzer detected trades successfully!")
        
        # Show detailed analysis
        total = trade_analysis.get('total', {})
        won = trade_analysis.get('won', {})
        lost = trade_analysis.get('lost', {})
        
        print(f"\\n📈 DETAILED ANALYSIS:")
        if 'total' in total:
            print(f"   Total trades: {total['total']}")
            print(f"   Total P&L: ${total.get('pnl', {}).get('net', 0):.2f}")
        
        if won and 'total' in won:
            print(f"   Winning trades: {won['total']}")
            print(f"   Average win: ${won.get('pnl', {}).get('average', 0):.2f}")
        
        if lost and 'total' in lost:
            print(f"   Losing trades: {lost['total']}")
            print(f"   Average loss: ${lost.get('pnl', {}).get('average', 0):.2f}")
            
        return True
    else:
        print("❌ TradeAnalyzer detected 0 trades - fix needed!")
        print(f"Raw trade analysis: {trade_analysis}")
        return False


if __name__ == "__main__":
    success = test_trade_analyzer()
    if success:
        print("\\n🎉 TradeAnalyzer test PASSED!")
    else:
        print("\\n❌ TradeAnalyzer test FAILED!")