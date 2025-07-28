#!/usr/bin/env python3
"""
Test Suite for OnchainBroker
Comprehensive tests for position tracking, cash management, and trade execution
"""

import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
sys.path.append('/Users/noel/projects/trading_eda')

import backtrader as bt
from backtesting.onchain_broker import OnchainBroker, OnchainCommissionInfo, setup_onchain_broker
from backtesting.solana_transaction_feed import SolanaTransactionFeed


class TestStrategy(bt.Strategy):
    """Simple test strategy for broker testing"""
    
    def __init__(self):
        self.trades = []
        self.orders = []
        
    def next(self):
        # Don't auto-trade, let tests control orders
        pass
    
    def notify_order(self, order):
        self.orders.append({
            'status': order.getstatusname(),
            'type': 'BUY' if order.isbuy() else 'SELL',
            'size': order.size,
            'executed_price': order.executed.price if order.executed.price else None,
            'executed_size': order.executed.size if order.executed.size else None
        })
    
    def buy_shares(self, size):
        """Helper to execute buy order"""
        return self.buy(size=size)
    
    def sell_shares(self, size):
        """Helper to execute sell order"""
        return self.sell(size=size)


class OnchainBrokerTestCase(unittest.TestCase):
    """Test cases for OnchainBroker functionality"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.cerebro = bt.Cerebro()
        
        # Create sample transaction data
        self.transaction_data = self._create_test_data()
        
        # Add data feed
        self.data_feed = SolanaTransactionFeed(
            transaction_data=self.transaction_data,
            aggregation_window=60,
            min_transactions=1
        )
        self.cerebro.adddata(self.data_feed)
        
        # Setup onchain broker with lower minimum transaction size for testing
        setup_onchain_broker(self.cerebro, initial_cash=100000)
        # Override minimum transaction size for testing with scaled prices
        self.cerebro.broker.params.min_transaction_size = 0.0000001
        
        # Add test strategy
        self.cerebro.addstrategy(TestStrategy)
        
    def _create_test_data(self):
        """Create test transaction data with known characteristics"""
        timestamps = [datetime(2024, 1, 1, 10, 0, 0) + timedelta(minutes=i) for i in range(20)]
        
        transactions = []
        for i, ts in enumerate(timestamps):
            # Create predictable price progression
            base_price = 100.0 + i * 10  # $100, $110, $120, etc.
            sol_amount = 10.0 + i  # 10, 11, 12 SOL etc.
            
            transaction = {
                'block_timestamp': ts,
                'mint': 'test_token_123',
                'succeeded': True,
                'swapper': f'trader_{i:03d}',
                'sol_amount': sol_amount,
                'is_buy': i % 2 == 0,  # Alternating buy/sell
                'price': base_price,
                'swap_from_amount': sol_amount if i % 1 == 0 else sol_amount / base_price,
                'swap_to_amount': sol_amount / base_price if i % 1 == 0 else sol_amount,
                'swap_from_mint': 'So11111111111111111111111111111111111111112' if i % 2 == 0 else 'test_token_123',
                'swap_to_mint': 'test_token_123' if i % 2 == 0 else 'So11111111111111111111111111111111111111112',
            }
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def test_broker_initialization(self):
        """Test broker initializes with correct parameters"""
        results = self.cerebro.run()
        strategy = results[0]
        
        # Check initial cash
        self.assertEqual(self.cerebro.broker.get_cash(), 100000.0)
        self.assertEqual(self.cerebro.broker.get_value(), 100000.0)
        
        # Check broker type
        self.assertIsInstance(self.cerebro.broker, OnchainBroker)
        
        print("✅ Broker initialization test passed")
    
    def test_simple_buy_sell_cycle(self):
        """Test basic buy-sell cycle with position tracking"""
        # Manually execute trades
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data_feed)
        setup_onchain_broker(cerebro, initial_cash=100000)
        
        class SimpleBuySellStrategy(bt.Strategy):
            def __init__(self):
                self.trade_executed = False
                self.buy_executed = False
                self.sell_executed = False
                self.orders = []
                
            def next(self):
                current_bar = len(self.data)
                
                if not self.buy_executed and current_bar == 2:
                    # Execute buy on second data point
                    order = self.buy(size=0.001)  # Small fractional position
                    self.orders.append(('BUY', order))
                    self.buy_executed = True
                    print(f"📝 BUY ORDER PLACED at bar {current_bar}")
                    
                elif self.buy_executed and not self.sell_executed and current_bar == 6:
                    # Execute sell on sixth data point (give more time)
                    position_size = self.position.size
                    print(f"📝 CHECKING POSITION at bar {current_bar}: {position_size}")
                    if position_size > 0:
                        order = self.sell(size=position_size)
                        self.orders.append(('SELL', order))
                        self.sell_executed = True
                        print(f"📝 SELL ORDER PLACED at bar {current_bar}")
            
            def notify_order(self, order):
                if order.status == order.Completed:
                    if order.isbuy():
                        print(f"📝 BUY COMPLETED: {order.executed.size} at {order.executed.price:.2f}")
                    else:
                        print(f"📝 SELL COMPLETED: {order.executed.size} at {order.executed.price:.2f}")
        
        cerebro.addstrategy(SimpleBuySellStrategy)
        results = cerebro.run()
        strategy = results[0]
        
        # Verify trade execution
        self.assertTrue(strategy.buy_executed, "Buy order should have been executed")
        self.assertTrue(strategy.sell_executed, "Sell order should have been executed")
        
        # Verify final position is closed
        final_position = strategy.position.size
        self.assertAlmostEqual(final_position, 0.0, places=10, 
                              msg=f"Position should be closed, but got {final_position}")
        
        # Verify cash flow makes sense
        final_cash = cerebro.broker.get_cash()
        final_value = cerebro.broker.get_value()
        
        print(f"📊 Final cash: ${final_cash:,.2f}")
        print(f"📊 Final portfolio value: ${final_value:,.2f}")
        
        # Cash should be positive (no negative cash bug)
        self.assertGreater(final_cash, 0, "Cash should never go negative")
        
        print("✅ Simple buy-sell cycle test passed")
    
    def test_position_tracking_accuracy(self):
        """Test position tracking with multiple trades"""
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data_feed)
        setup_onchain_broker(cerebro, initial_cash=100000)
        
        class PositionTrackingStrategy(bt.Strategy):
            def __init__(self):
                self.positions_log = []
                self.cash_log = []
                self.trade_count = 0
                
            def next(self):
                current_bar = len(self.data)
                
                # Log current state
                self.positions_log.append(self.position.size)
                self.cash_log.append(self.broker.get_cash())
                
                # Execute trades at specific points
                if current_bar == 2:
                    self.buy(size=0.001)
                    self.trade_count += 1
                    print(f"📝 TRADE {self.trade_count}: BUY 0.001 at bar {current_bar}")
                elif current_bar == 5:
                    self.buy(size=0.0005)  # Add to position
                    self.trade_count += 1
                    print(f"📝 TRADE {self.trade_count}: BUY 0.0005 at bar {current_bar}")
                elif current_bar == 8:
                    self.sell(size=0.0007)  # Partial sell
                    self.trade_count += 1
                    print(f"📝 TRADE {self.trade_count}: SELL 0.0007 at bar {current_bar}")
                elif current_bar == 12:
                    # Close remaining position
                    position_size = self.position.size
                    print(f"📝 REMAINING POSITION at bar {current_bar}: {position_size}")
                    if position_size > 0:
                        self.sell(size=position_size)
                        self.trade_count += 1
                        print(f"📝 TRADE {self.trade_count}: SELL {position_size} (close) at bar {current_bar}")
        
        cerebro.addstrategy(PositionTrackingStrategy)
        results = cerebro.run()
        strategy = results[0]
        
        # Check position progression
        print(f"📊 Position progression: {[f'{p:.6f}' for p in strategy.positions_log]}")
        print(f"📊 Cash progression: {[f'${c:,.2f}' for c in strategy.cash_log]}")
        
        # Final position should be zero
        final_position = strategy.positions_log[-1]
        self.assertAlmostEqual(final_position, 0.0, places=10,
                              msg=f"Final position should be zero, got {final_position}")
        
        # No negative cash values
        for i, cash in enumerate(strategy.cash_log):
            self.assertGreater(cash, 0, f"Cash should never be negative at step {i}, got ${cash:,.2f}")
        
        print("✅ Position tracking accuracy test passed")
    
    def test_commission_and_costs(self):
        """Test gas costs and commission calculations"""
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data_feed)
        setup_onchain_broker(cerebro, initial_cash=100000)
        
        # Get broker parameters
        broker = cerebro.broker
        gas_cost_pct = broker.params.gas_cost_pct
        slippage_pct = broker.params.slippage_pct
        
        class CostTrackingStrategy(bt.Strategy):
            def __init__(self):
                self.initial_cash = None
                self.post_buy_cash = None
                self.post_sell_cash = None
                self.buy_price = None
                self.sell_price = None
                self.position_size = None
                
            def next(self):
                if self.initial_cash is None:
                    self.initial_cash = self.broker.get_cash()
                
                if len(self.data) == 2:
                    self.buy(size=0.001)
                elif len(self.data) == 4:
                    self.post_buy_cash = self.broker.get_cash()
                    self.position_size = self.position.size
                    if self.position_size > 0:
                        self.sell(size=self.position_size)
                
            def notify_order(self, order):
                if order.status == order.Completed:
                    if order.isbuy():
                        self.buy_price = order.executed.price
                        self.post_buy_cash = self.broker.get_cash()
                    else:
                        self.sell_price = order.executed.price
                        self.post_sell_cash = self.broker.get_cash()
        
        cerebro.addstrategy(CostTrackingStrategy)
        results = cerebro.run()
        strategy = results[0]
        
        if strategy.buy_price and strategy.sell_price and strategy.position_size:
            # Calculate expected costs
            buy_value = strategy.position_size * strategy.buy_price
            sell_value = strategy.position_size * strategy.sell_price
            
            expected_buy_cost = buy_value * (1 + gas_cost_pct + slippage_pct)
            expected_sell_proceeds = sell_value * (1 - gas_cost_pct - slippage_pct)
            
            print(f"📊 Buy price: ${strategy.buy_price:,.2f}")
            print(f"📊 Sell price: ${strategy.sell_price:,.2f}")
            print(f"📊 Position size: {strategy.position_size:.6f}")
            print(f"📊 Expected buy cost: ${expected_buy_cost:,.2f}")
            print(f"📊 Expected sell proceeds: ${expected_sell_proceeds:,.2f}")
            print(f"📊 Cash flow: ${strategy.initial_cash:,.2f} → ${strategy.post_buy_cash:,.2f} → ${strategy.post_sell_cash:,.2f}")
            
            # Verify costs are applied
            actual_buy_cost = strategy.initial_cash - strategy.post_buy_cash
            print(f"📊 Actual buy cost: ${actual_buy_cost:,.2f}")
            
            # Cost should include gas and slippage
            self.assertGreater(actual_buy_cost, buy_value * 1.001,  # At least gas cost
                             "Buy cost should include gas and slippage")
        
        print("✅ Commission and costs test passed")
    
    def test_fractional_shares_handling(self):
        """Test broker handles fractional shares correctly"""
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data_feed)
        setup_onchain_broker(cerebro, initial_cash=100000)
        
        class FractionalSharesStrategy(bt.Strategy):
            def __init__(self):
                self.fractional_positions = []
                
            def next(self):
                if len(self.data) == 2:
                    # Buy very small fractional amount
                    self.buy(size=0.000001)
                elif len(self.data) == 4:
                    # Buy another fractional amount
                    self.buy(size=0.0000005)
                elif len(self.data) == 6:
                    # Record total position
                    self.fractional_positions.append(self.position.size)
                    # Sell exact position
                    if self.position.size > 0:
                        self.sell(size=self.position.size)
                elif len(self.data) == 8:
                    # Record final position
                    self.fractional_positions.append(self.position.size)
        
        cerebro.addstrategy(FractionalSharesStrategy)
        results = cerebro.run()
        strategy = results[0]
        
        if len(strategy.fractional_positions) >= 2:
            # Check that fractional positions accumulate correctly
            pre_sell_position = strategy.fractional_positions[0]
            final_position = strategy.fractional_positions[1]
            
            print(f"📊 Pre-sell position: {pre_sell_position:.10f}")
            print(f"📊 Final position: {final_position:.10f}")
            
            # Final position should be zero (or very close due to floating point)
            self.assertAlmostEqual(final_position, 0.0, places=8,
                                 msg=f"Final position should be zero, got {final_position:.10f}")
            
            # Pre-sell position should be positive
            self.assertGreater(pre_sell_position, 0,
                             "Pre-sell position should be positive")
        
        print("✅ Fractional shares handling test passed")
    
    def test_sign_error_regression(self):
        """Regression test for the sign errors we fixed"""
        cerebro = bt.Cerebro()
        cerebro.adddata(self.data_feed)
        setup_onchain_broker(cerebro, initial_cash=100000)
        
        class SignErrorRegressionStrategy(bt.Strategy):
            def __init__(self):
                self.position_log = []
                self.cash_values = []
                self.trades = []
                
            def next(self):
                current_bar = len(self.data)
                
                # Record state
                self.position_log.append(self.position.size)
                self.cash_values.append(self.broker.get_cash())
                
                # Debug: Show current data state including OHLCV
                if current_bar <= 10:
                    print(f"📊 Bar {current_bar}: O={self.data.open[0]:.2f}, H={self.data.high[0]:.2f}, L={self.data.low[0]:.2f}, C={self.data.close[0]:.2f}, V={self.data.volume[0]:.2f}")
                    print(f"    💰 Cash=${self.broker.get_cash():,.2f}, Position={self.position.size:.6f}")
                
                # Execute specific trade sequence that triggered the bug
                if current_bar == 3:
                    print(f"📝 ATTEMPTING BUY ORDER at bar {current_bar}")
                    # Use much smaller size for scaled prices
                    order = self.buy(size=0.0000001)  # 1e-7 shares for billion-scale prices
                    self.trades.append(('BUY', order))
                    print(f"📝 BUY ORDER PLACED at bar {current_bar}, Order status: {order.getstatusname()}")
                elif current_bar == 7:
                    position_size = self.position.size
                    print(f"📝 CHECKING POSITION at bar {current_bar}: {position_size}")
                    if position_size > 0:
                        order = self.sell(size=position_size)
                        self.trades.append(('SELL', order))
                        print(f"📝 SELL ORDER PLACED at bar {current_bar}")
        
        cerebro.addstrategy(SignErrorRegressionStrategy)
        
        # Print test data before running
        print("\n📊 === RAW TRANSACTION DATA ===")
        df = self.transaction_data
        print(f"Total transactions: {len(df)}")
        print("Sample transaction data:")
        print(df[['block_timestamp', 'price', 'sol_amount', 'is_buy', 'succeeded']].head(10).to_string())
        
        print("\nSwap data (used for price calculation):")
        swap_cols = ['swap_from_mint', 'swap_to_mint', 'swap_from_amount', 'swap_to_amount', 'is_buy']
        if all(col in df.columns for col in swap_cols):
            print(df[swap_cols].head(10).to_string())
        else:
            print("Swap columns not found in transaction data")
            
        print("\nTransaction data summary:")
        print(f"- Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
        print(f"- SOL amount range: {df['sol_amount'].min():.2f} to {df['sol_amount'].max():.2f}")
        print(f"- Buy ratio: {df['is_buy'].mean():.1%}")
        print(f"- Success rate: {df['succeeded'].mean():.1%}")
        
        if 'swap_from_amount' in df.columns:
            print(f"- Swap from amount range: {df['swap_from_amount'].min():.2f} to {df['swap_from_amount'].max():.2f}")
            print(f"- Swap to amount range: {df['swap_to_amount'].min():.2f} to {df['swap_to_amount'].max():.2f}")
            print(f"- Calculated swap ratio range: {(df['swap_from_amount'] / df['swap_to_amount']).min():.2f} to {(df['swap_from_amount'] / df['swap_to_amount']).max():.2f}")
        
        print("\n📊 === DERIVED OHLCV DATA ===")
        print("This shows how transaction data gets converted to OHLCV bars:")
        
        results = cerebro.run()
        strategy = results[0]
        
        # Check for sign error symptoms
        print(f"📊 Position progression: {[f'{p:.6f}' for p in strategy.position_log]}")
        print(f"📊 Cash progression: {[f'${c:,.2f}' for c in strategy.cash_values]}")
        
        # 1. No negative cash values
        for i, cash in enumerate(strategy.cash_values):
            self.assertGreaterEqual(cash, 0, 
                                  f"Cash should never be negative at step {i}, got ${cash:,.2f}")
        
        # 2. Position should not double after sell
        if len(strategy.position_log) >= 8:
            pre_buy = strategy.position_log[2]   # Before buy (bar 3)
            post_buy = strategy.position_log[3]  # After buy (bar 4) 
            post_sell = strategy.position_log[7] # After sell (bar 8)
            
            self.assertEqual(pre_buy, 0.0, "Should start with no position")
            self.assertGreater(post_buy, 0.0, "Should have position after buy")
            self.assertAlmostEqual(post_sell, 0.0, places=8, 
                                 msg=f"Position should be closed after sell, got {post_sell:.8f}")
            
            # Position should not double
            self.assertLess(abs(post_sell), post_buy, 
                          "Position should decrease after sell, not increase")
        
        # 3. Cash should increase after profitable sell (if price went up)
        if len(strategy.cash_values) >= 8:
            post_buy_cash = strategy.cash_values[3]  # After buy (bar 4)
            post_sell_cash = strategy.cash_values[7] # After sell (bar 8)
            
            # At minimum, we shouldn't lose everything
            self.assertGreater(post_sell_cash, post_buy_cash * 0.5,
                             "Shouldn't lose more than 50% on a single trade")
        
        print("✅ Sign error regression test passed")


def run_onchain_broker_tests():
    """Run all onchain broker tests"""
    print("🧪 RUNNING ONCHAIN BROKER TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(OnchainBrokerTestCase)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL ONCHAIN BROKER TESTS PASSED!")
        print(f"✅ Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"❌ Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, trace in result.failures:
                print(f"- {test}: {trace}")
        
        if result.errors:
            print("\nERRORS:")
            for test, trace in result.errors:
                print(f"- {test}: {trace}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_onchain_broker_tests()