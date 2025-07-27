#!/usr/bin/env python3
"""
Full Trading Session Precision Test
Tests numerical precision stability over extended trading sessions with scaled prices.
"""

import sys
import os
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import backtrader as bt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_transaction_feed import SolanaTransactionFeed


class PrecisionTestStrategy(bt.Strategy):
    """
    Strategy designed to stress-test numerical precision over long sessions.
    
    Strategy:
    1. Make frequent small trades to test cumulative precision
    2. Track precision drift in portfolio value calculations
    3. Test with varying position sizes (small to large)
    4. Monitor for precision degradation over time
    """
    
    params = (
        ('trade_frequency', 5),      # Trade every N bars
        ('position_sizes', [0.1, 1.0, 10.0]),  # SOL amounts to test
    )
    
    def __init__(self):
        self.precision_log = []
        self.trade_count = 0
        self.total_trades = 0
        self.cumulative_pnl = 0.0
        self.price_scale_factor = self.data.PRICE_SCALE_FACTOR
        self.bars_processed = 0
        self.current_position_size_idx = 0
        
        # Track precision metrics
        self.initial_cash = None
        self.portfolio_value_history = []
        self.pnl_history = []
        self.precision_errors = []
        
        print(f"🧮 Precision Test Strategy initialized")
        print(f"⚖️  Scale factor: {self.price_scale_factor:,.0f}")
        print(f"🎯 Test position sizes: {self.params.position_sizes} SOL")
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def next(self):
        """Called for each data point"""
        self.bars_processed += 1
        current_price = self.data.close[0]
        current_time = self.data.datetime.datetime(0)
        
        # Initialize cash tracking
        if self.initial_cash is None:
            self.initial_cash = self.broker.getvalue()
            print(f"💰 Initial cash: {self.initial_cash:,.2f} scaled = {self.initial_cash/self.price_scale_factor:.6f} SOL")
        
        # Record portfolio value for precision tracking
        portfolio_value = self.broker.getvalue()
        self.portfolio_value_history.append(portfolio_value)
        
        # Log precision metrics every 10 bars
        if self.bars_processed % 10 == 0:
            self._log_precision_metrics(current_time, current_price, portfolio_value)
        
        # Execute trades at specified frequency
        if self.bars_processed % self.params.trade_frequency == 0:
            self._execute_precision_test_trade(current_price)
    
    def _log_precision_metrics(self, current_time, current_price, portfolio_value):
        """Log detailed precision metrics"""
        position_value_scaled = self.position.size * current_price if self.position.size > 0 else 0
        position_value_sol = position_value_scaled / self.price_scale_factor
        portfolio_value_sol = portfolio_value / self.price_scale_factor
        
        # Calculate precision indicators
        if len(self.portfolio_value_history) > 1:
            value_change = portfolio_value - self.portfolio_value_history[-2]
            value_change_sol = value_change / self.price_scale_factor
        else:
            value_change = 0
            value_change_sol = 0
        
        self.log(f'📊 Bar {self.bars_processed}: Price {current_price:.6f}, '
                f'Portfolio {portfolio_value_sol:.8f} SOL, '
                f'Position {self.position.size:,.0f} tokens ({position_value_sol:.8f} SOL)')
        
        # Store precision data
        precision_data = {
            'bar': self.bars_processed,
            'timestamp': current_time,
            'price_scaled': current_price,
            'price_sol': current_price / self.price_scale_factor,
            'portfolio_scaled': portfolio_value,
            'portfolio_sol': portfolio_value_sol,
            'position_tokens': self.position.size,
            'position_value_sol': position_value_sol,
            'value_change_sol': value_change_sol
        }
        self.precision_log.append(precision_data)
    
    def _execute_precision_test_trade(self, current_price):
        """Execute trades to test precision under various scenarios"""
        if current_price <= 0:
            return
        
        # Cycle through different position sizes
        target_sol = self.params.position_sizes[self.current_position_size_idx]
        self.current_position_size_idx = (self.current_position_size_idx + 1) % len(self.params.position_sizes)
        
        if not self.position:
            # Buy with current target size
            tokens_to_buy = (target_sol * self.price_scale_factor) / current_price
            self.log(f'🛒 BUY {tokens_to_buy:,.0f} tokens (target: {target_sol} SOL)')
            self.buy(size=tokens_to_buy)
            
        else:
            # Sell current position
            self.log(f'🔔 SELL {self.position.size:,.0f} tokens')
            self.sell(size=self.position.size)
            self.total_trades += 1
    
    def notify_trade(self, trade):
        """Track trade precision"""
        if not trade.isclosed:
            return
        
        pnl_scaled = trade.pnl
        pnl_sol = pnl_scaled / self.price_scale_factor
        self.cumulative_pnl += pnl_sol
        self.pnl_history.append(pnl_sol)
        
        # Test for precision issues
        precision_error = self._detect_precision_error(pnl_sol, pnl_scaled)
        if precision_error:
            self.precision_errors.append(precision_error)
        
        self.log(f'💰 Trade closed: P&L {pnl_sol:+.8f} SOL (Cumulative: {self.cumulative_pnl:+.8f} SOL)')
    
    def _detect_precision_error(self, pnl_sol, pnl_scaled):
        """Detect potential precision errors in calculations"""
        # Check if conversion back to SOL matches expected
        expected_pnl_sol = pnl_scaled / self.price_scale_factor
        conversion_error = abs(pnl_sol - expected_pnl_sol)
        
        if conversion_error > 1e-12:  # Threshold for precision error
            return {
                'type': 'conversion_error',
                'pnl_sol': pnl_sol,
                'pnl_scaled': pnl_scaled,
                'expected_sol': expected_pnl_sol,
                'error': conversion_error
            }
        
        # Check for unrealistic P&L values
        if abs(pnl_sol) > 1000:  # >1000 SOL profit/loss seems unrealistic for test
            return {
                'type': 'unrealistic_pnl',
                'pnl_sol': pnl_sol,
                'pnl_scaled': pnl_scaled
            }
        
        return None
    
    def stop(self):
        """Analyze precision results at end of session"""
        final_portfolio = self.broker.getvalue()
        final_portfolio_sol = final_portfolio / self.price_scale_factor
        initial_sol = (self.initial_cash or final_portfolio) / self.price_scale_factor
        
        print(f"\n📊 PRECISION TEST RESULTS")
        print("=" * 60)
        print(f"⏱️  Bars processed: {self.bars_processed}")
        print(f"🔄 Total trades: {self.total_trades}")
        print(f"💰 Initial: {initial_sol:.8f} SOL")
        print(f"💰 Final: {final_portfolio_sol:.8f} SOL")
        print(f"📈 Total P&L: {final_portfolio_sol - initial_sol:+.8f} SOL")
        print(f"🧮 Cumulative trade P&L: {self.cumulative_pnl:+.8f} SOL")
        
        # Precision analysis
        self._analyze_precision_stability()
        
        # Store results for external analysis
        self.final_results = {
            'bars_processed': self.bars_processed,
            'total_trades': self.total_trades,
            'initial_cash_sol': initial_sol,
            'final_portfolio_sol': final_portfolio_sol,
            'total_pnl_sol': final_portfolio_sol - initial_sol,
            'cumulative_trade_pnl': self.cumulative_pnl,
            'precision_errors': self.precision_errors,
            'precision_log': self.precision_log
        }
    
    def _analyze_precision_stability(self):
        """Analyze numerical precision stability over the session"""
        if len(self.precision_log) < 2:
            print("⚠️  Insufficient data for precision analysis")
            return
        
        # Check for precision drift in portfolio values
        portfolio_values = [entry['portfolio_sol'] for entry in self.precision_log]
        
        # Look for unrealistic jumps in portfolio value
        large_jumps = []
        for i in range(1, len(portfolio_values)):
            change = abs(portfolio_values[i] - portfolio_values[i-1])
            if change > 100:  # >100 SOL change in one bar seems unrealistic
                large_jumps.append({
                    'bar': self.precision_log[i]['bar'],
                    'change': change,
                    'from': portfolio_values[i-1],
                    'to': portfolio_values[i]
                })
        
        # Check for precision in small price movements
        prices = [entry['price_sol'] for entry in self.precision_log]
        small_price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices)) 
                              if abs(prices[i] - prices[i-1]) < 1e-10]
        
        print(f"\n🔍 PRECISION ANALYSIS:")
        print(f"   Large portfolio jumps: {len(large_jumps)}")
        if large_jumps:
            for jump in large_jumps[:3]:  # Show first 3
                print(f"     Bar {jump['bar']}: {jump['from']:.8f} → {jump['to']:.8f} SOL")
        
        print(f"   Small price changes detected: {len(small_price_changes)}")
        print(f"   Precision errors found: {len(self.precision_errors)}")
        
        if self.precision_errors:
            print(f"   Error types: {[e['type'] for e in self.precision_errors]}")
        
        # Overall assessment
        if len(self.precision_errors) == 0 and len(large_jumps) == 0:
            print("✅ PRECISION TEST PASSED: No significant precision issues detected")
        elif len(self.precision_errors) < 5 and len(large_jumps) < 3:
            print("⚠️  PRECISION TEST PARTIAL: Minor precision issues detected")
        else:
            print("❌ PRECISION TEST FAILED: Significant precision issues detected")


def test_session_precision():
    """Test numerical precision over a full trading session."""
    print("🧪 Testing Numerical Precision Over Full Trading Session")
    print("=" * 60)
    
    # Connect to DuckDB and get data for a long session
    db_path = '/Volumes/Extreme SSD/DuckDB/solana.duckdb'
    conn = duckdb.connect(db_path)
    
    # Use known working coin for extended precision testing
    coin_id = "8xPe3DMr52oYAkCBk57ZeQE1h5zBkWNk37eE4J8spump"
    print(f"📊 Using known working coin: {coin_id}")
    
    # Get extended transaction data
    print(f"\n1. Loading extended transaction data...")
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
    LIMIT 1000  -- Limit for reasonable test duration
    """
    
    transaction_df = conn.execute(transaction_query).fetchdf()
    conn.close()
    
    print(f"✅ Loaded {len(transaction_df)} transactions")
    print(f"   Date range: {transaction_df['block_timestamp'].min()} to {transaction_df['block_timestamp'].max()}")
    
    # Create feed for extended session
    print(f"\n2. Creating extended session feed...")
    feed = SolanaTransactionFeed(transaction_data=transaction_df)
    
    print(f"✅ Feed created with {len(feed.sample_timestamps)} sample periods")
    
    # Set up Backtrader for extended session
    print(f"\n3. Running extended precision test...")
    cerebro = bt.Cerebro()
    
    # Add data feed
    cerebro.adddata(feed)
    
    # Add precision test strategy
    cerebro.addstrategy(PrecisionTestStrategy, trade_frequency=3, position_sizes=[0.5, 2.0, 10.0])
    
    # Set large initial cash for extended testing
    initial_sol = 1000.0  # 1000 SOL for extended testing
    initial_scaled_cash = initial_sol * feed.PRICE_SCALE_FACTOR
    cerebro.broker.setcash(initial_scaled_cash)
    
    # Add commission
    cerebro.broker.setcommission(commission=0.001)
    
    # Run the extended session test
    print("=" * 60)
    result = cerebro.run()
    
    # Analyze results
    strategy = result[0]
    final_results = strategy.final_results
    
    print(f"\n📈 EXTENDED SESSION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Session duration: {len(feed.sample_timestamps)} sample periods")
    print(f"🔄 Total trades executed: {final_results['total_trades']}")
    print(f"💰 Portfolio drift: {final_results['total_pnl_sol']:+.8f} SOL")
    print(f"🧮 Scale factor tested: {feed.PRICE_SCALE_FACTOR:,.0f}")
    print(f"⚠️  Precision errors: {len(final_results['precision_errors'])}")
    
    # Final assessment
    if len(final_results['precision_errors']) == 0:
        print(f"\n🎉 EXTENDED SESSION PRECISION TEST PASSED!")
        print(f"   ✅ No precision errors over {final_results['bars_processed']} bars")
        print(f"   ✅ {final_results['total_trades']} trades executed cleanly")
        print(f"   ✅ Portfolio tracking remained accurate")
    else:
        print(f"\n⚠️  EXTENDED SESSION PRECISION TEST - ISSUES DETECTED")
        print(f"   Found {len(final_results['precision_errors'])} precision issues")
        for error in final_results['precision_errors'][:3]:  # Show first 3
            print(f"   - {error['type']}: {error}")


if __name__ == '__main__':
    test_session_precision()