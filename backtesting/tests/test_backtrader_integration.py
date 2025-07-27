#!/usr/bin/env python3
"""
Backtrader Integration Test for Scaled Price Feed
Tests order execution, P&L calculation, and position management with scaled prices.
"""

import sys
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import backtrader as bt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_transaction_feed import SolanaTransactionFeed


class ScaledPriceTestStrategy(bt.Strategy):
    """
    Simple test strategy to validate scaled price handling.
    
    Strategy:
    1. Buy 1 SOL worth of tokens at market price
    2. Hold for 3 sample periods  
    3. Sell entire position
    4. Validate P&L calculations
    """
    
    params = (
        ('target_sol_amount', 1.0),  # Target SOL amount to invest
        ('hold_periods', 3),         # How many periods to hold
    )
    
    def __init__(self):
        self.order = None
        self.position_entry_price = None
        self.position_entry_time = None
        self.periods_held = 0
        self.max_position_value = 0
        self.trades_completed = 0
        
        # Track price scaling factor
        self.price_scale_factor = self.data.PRICE_SCALE_FACTOR
        
        print(f"💰 Strategy initialized with price scale factor: {self.price_scale_factor:,.0f}")
        print(f"🎯 Target investment: {self.params.target_sol_amount} SOL")
        print(f"⏱️  Hold periods: {self.params.hold_periods}")
    
    def log(self, txt, dt=None):
        """Logging function for the strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def next(self):
        """Called for each data point"""
        current_time = self.data.datetime.datetime(0)
        
        # Get all OHLCV data (scaled prices)
        open_price = self.data.open[0]
        high_price = self.data.high[0]
        low_price = self.data.low[0]
        close_price = self.data.close[0]
        volume = self.data.volume[0]
        
        # Convert scaled prices back to SOL for display
        open_sol = open_price / self.price_scale_factor
        high_sol = high_price / self.price_scale_factor
        low_sol = low_price / self.price_scale_factor
        close_sol = close_price / self.price_scale_factor
        
        self.log(f'OHLCV: O:{open_price:6.2f} H:{high_price:6.2f} L:{low_price:6.2f} C:{close_price:6.2f} V:{volume:6.2f}')
        self.log(f'  SOL: O:{open_sol:.10f} H:{high_sol:.10f} L:{low_sol:.10f} C:{close_sol:.10f}')
        self.log(f'Position: {self.position.size:,.0f} tokens')
        
        # Track maximum position value
        if self.position.size > 0:
            position_value_sol = (self.position.size * close_sol)
            self.max_position_value = max(self.max_position_value, position_value_sol)
        
        # Check if we have pending orders
        if self.order:
            return
        
        # Trading logic
        if not self.position:
            # No position - try to buy
            if close_price > 0:  # Valid price available
                # Calculate how many tokens we can buy with target SOL amount
                tokens_to_buy = (self.params.target_sol_amount * self.price_scale_factor) / close_price
                
                self.log(f'🛒 BUYING {tokens_to_buy:,.0f} tokens at close price {close_price:.2f}')
                self.log(f'   Expected SOL cost: {self.params.target_sol_amount:.6f}')
                
                # Place buy order
                self.order = self.buy(size=tokens_to_buy)
                self.position_entry_price = close_price
                self.position_entry_time = current_time
                
        else:
            # We have a position - check if it's time to sell
            self.periods_held += 1
            
            if self.periods_held >= self.params.hold_periods:
                self.log(f'🔔 SELLING entire position of {self.position.size:,.0f} tokens')
                self.log(f'   Current close price: {close_price:.2f} scaled ({close_sol:.10f} SOL/token)')
                
                # Place sell order for entire position
                self.order = self.sell(size=self.position.size)
    
    def notify_order(self, order):
        """Called when order status changes"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                actual_cost_sol = (order.executed.size * order.executed.price) / self.price_scale_factor
                self.log(f'✅ BUY EXECUTED: {order.executed.size:,.0f} tokens at {order.executed.price:.2f} scaled')
                self.log(f'   Actual SOL cost: {actual_cost_sol:.6f} (vs target {self.params.target_sol_amount:.6f})')
                self.log(f'   Cost difference: {abs(actual_cost_sol - self.params.target_sol_amount):.6f} SOL')
                
            elif order.issell():
                actual_proceeds_sol = (order.executed.size * order.executed.price) / self.price_scale_factor
                self.log(f'✅ SELL EXECUTED: {order.executed.size:,.0f} tokens at {order.executed.price:.2f} scaled')
                self.log(f'   SOL proceeds: {actual_proceeds_sol:.6f}')
                
                # Calculate P&L
                if self.position_entry_price:
                    entry_cost_sol = (order.executed.size * self.position_entry_price) / self.price_scale_factor
                    pnl_sol = actual_proceeds_sol - entry_cost_sol
                    pnl_percentage = (pnl_sol / entry_cost_sol) * 100
                    
                    self.log(f'💰 TRADE P&L: {pnl_sol:+.6f} SOL ({pnl_percentage:+.2f}%)')
                    self.log(f'   Entry cost: {entry_cost_sol:.6f} SOL at {self.position_entry_price:.2f} scaled')
                    self.log(f'   Exit proceeds: {actual_proceeds_sol:.6f} SOL at {order.executed.price:.2f} scaled')
                
                self.trades_completed += 1
                self.periods_held = 0
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'❌ Order {order.status}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Called when a trade is closed"""
        if not trade.isclosed:
            return
        
        # Convert P&L back to SOL units for validation
        scaled_pnl = trade.pnl
        sol_pnl = scaled_pnl / self.price_scale_factor
        
        self.log(f'🎯 TRADE CLOSED: P&L {scaled_pnl:+.2f} scaled = {sol_pnl:+.8f} SOL')
        self.log(f'   Commission: {trade.pnlcomm:+.2f} scaled')
    
    def stop(self):
        """Called when strategy ends"""
        portfolio_value_scaled = self.broker.getvalue()
        portfolio_value_sol = portfolio_value_scaled / self.price_scale_factor
        
        print(f"\n📊 STRATEGY PERFORMANCE SUMMARY")
        print(f"=" * 60)
        print(f"💼 Final portfolio value: {portfolio_value_scaled:,.2f} scaled = {portfolio_value_sol:.6f} SOL")
        print(f"🏆 Trades completed: {self.trades_completed}")
        print(f"📈 Max position value: {self.max_position_value:.6f} SOL")
        print(f"⚖️  Price scale factor: {self.price_scale_factor:,.0f}")


def test_backtrader_integration():
    """Test Backtrader integration with scaled prices using real transaction data."""
    print("🧪 Testing Backtrader Integration with Scaled Prices")
    print("=" * 60)
    
    # Connect to DuckDB and get real data
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Test coin
    coin_id = "8xPe3DMr52oYAkCBk57ZeQE1h5zBkWNk37eE4J8spump"
    print(f"Testing coin: {coin_id}")
    
    # Get real transaction data
    print("\n1. Loading real transaction data...")
    transaction_query = f"""
    SELECT 
        block_timestamp,
        mint,
        swapper,
        succeeded,
        swap_from_mint,
        swap_to_mint,
        swap_from_amount,
        swap_to_amount,
        CASE WHEN mint = swap_to_mint THEN 1 ELSE 0 END as is_buy,
        CASE 
            WHEN mint = swap_to_mint AND swap_from_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_from_amount
            WHEN mint = swap_from_mint AND swap_to_mint = 'So11111111111111111111111111111111111111112' 
            THEN swap_to_amount
            ELSE 0.0
        END as sol_amount
    FROM first_day_trades
    WHERE mint = '{coin_id}'
    AND succeeded = TRUE
    AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
         OR swap_to_mint = 'So11111111111111111111111111111111111111112')
    AND mint != 'So11111111111111111111111111111111111111112'
    ORDER BY block_timestamp
    LIMIT 500
    """
    
    transaction_df = conn.execute(transaction_query).fetchdf()
    conn.close()
    
    if len(transaction_df) == 0:
        print("❌ No transaction data found")
        return
    
    print(f"✅ Loaded {len(transaction_df)} transactions")
    
    # Create SolanaTransactionFeed
    print("\n2. Creating SolanaTransactionFeed...")
    feed = SolanaTransactionFeed(transaction_data=transaction_df)
    
    # Create Backtrader Cerebro
    print("\n3. Setting up Backtrader...")
    cerebro = bt.Cerebro()
    
    # Add our data feed
    cerebro.adddata(feed)
    
    # Add strategy
    cerebro.addstrategy(ScaledPriceTestStrategy, target_sol_amount=1.0, hold_periods=3)
    
    # Set initial cash (in scaled units - 100 SOL = 100 * 1e9 scaled units)
    initial_sol = 100.0
    initial_scaled_cash = initial_sol * feed.PRICE_SCALE_FACTOR
    cerebro.broker.setcash(initial_scaled_cash)
    
    print(f"💰 Initial cash: {initial_scaled_cash:,.0f} scaled = {initial_sol} SOL")
    
    # Add commission (as percentage of trade value)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Run backtest
    print(f"\n4. Running backtest with {len(feed.sample_timestamps)} sample periods...")
    print("=" * 60)
    
    result = cerebro.run()
    
    # Performance summary
    final_value_scaled = cerebro.broker.getvalue()
    final_value_sol = final_value_scaled / feed.PRICE_SCALE_FACTOR
    total_pnl_sol = final_value_sol - initial_sol
    
    print(f"\n📈 BACKTEST RESULTS")
    print("=" * 60)
    print(f"💰 Initial cash: {initial_sol:.6f} SOL")
    print(f"💰 Final value: {final_value_sol:.6f} SOL")
    print(f"📊 Total P&L: {total_pnl_sol:+.6f} SOL ({(total_pnl_sol/initial_sol)*100:+.2f}%)")
    print(f"⚙️  Price scaling factor: {feed.PRICE_SCALE_FACTOR:,.0f}")
    
    # Validation checks
    print(f"\n🔍 VALIDATION CHECKS")
    print("=" * 30)
    
    # Check if final value makes sense
    if abs(final_value_sol) < 1e-6:
        print("❌ Final portfolio value is essentially zero - possible precision issue")
    elif final_value_sol < 0:
        print("❌ Negative portfolio value - possible calculation error")
    else:
        print("✅ Portfolio value is positive and reasonable")
    
    # Check price scaling consistency
    if feed.PRICE_SCALE_FACTOR == 1e9:
        print("✅ Price scale factor is correct (1e9)")
    else:
        print(f"⚠️  Unexpected price scale factor: {feed.PRICE_SCALE_FACTOR}")
    
    # Check if trades occurred
    strategy = result[0]
    if strategy.trades_completed > 0:
        print(f"✅ {strategy.trades_completed} trades completed successfully")
    else:
        print("⚠️  No trades completed - check strategy logic")
    
    print(f"\n🎉 Backtrader integration test completed!")


if __name__ == '__main__':
    test_backtrader_integration()