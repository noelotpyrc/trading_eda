#!/usr/bin/env python3
"""
Unit tests for transaction-based price calculation in SolanaTransactionFeed.
Tests the price calculation logic and OHLCV aggregation functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solana_transaction_feed import SolanaTransactionFeed


class TestPriceCalculation(unittest.TestCase):
    """Test price calculation functionality."""
    
    def setUp(self):
        """Set up test data."""
        # SOL mint address
        self.SOL_MINT = 'So11111111111111111111111111111111111111112'
        self.TOKEN_MINT = 'TokenMint123456789'
        
        # Base timestamp for tests
        self.base_time = datetime(2024, 1, 1, 12, 0, 0)
        
    def create_test_transaction(self, timestamp, swap_from_mint, swap_to_mint, 
                              swap_from_amount, swap_to_amount, succeeded=True):
        """Helper to create test transaction data."""
        return {
            'block_timestamp': timestamp,
            'swap_from_mint': swap_from_mint,
            'swap_to_mint': swap_to_mint,
            'swap_from_amount': swap_from_amount,
            'swap_to_amount': swap_to_amount,
            'succeeded': succeeded,
            'is_buy': 1 if swap_from_mint == self.SOL_MINT else 0,
            'sol_amount': swap_from_amount if swap_from_mint == self.SOL_MINT else swap_to_amount,
            'swapper': 'test_wallet_123'
        }
    
    def test_sol_to_token_price_calculation(self):
        """Test price calculation for SOL → Token trades (with 1e9 scaling)."""
        # Create test transaction: 10 SOL → 1000 tokens
        # Expected raw price: 10 / 1000 = 0.01 SOL per token
        # Expected scaled price: 0.01 * 1e9 = 10,000,000
        transaction_data = pd.DataFrame([
            self.create_test_transaction(
                timestamp=self.base_time,
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=10.0,
                swap_to_amount=1000.0
            )
        ])
        
        # Create feed and calculate prices
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        # Check calculated price (scaled)
        raw_price = 10.0 / 1000.0  # 0.01
        expected_scaled_price = raw_price * feed.PRICE_SCALE_FACTOR  # 0.01 * 1e9 = 10,000,000
        actual_price = feed.df['transaction_price'].iloc[0]
        
        self.assertAlmostEqual(actual_price, expected_scaled_price, places=6)
        
    def test_token_to_sol_price_calculation(self):
        """Test price calculation for Token → SOL trades (with 1e9 scaling)."""
        # Create test transaction: 500 tokens → 5 SOL
        # Expected raw price: 5 / 500 = 0.01 SOL per token
        # Expected scaled price: 0.01 * 1e9 = 10,000,000
        transaction_data = pd.DataFrame([
            self.create_test_transaction(
                timestamp=self.base_time,
                swap_from_mint=self.TOKEN_MINT,
                swap_to_mint=self.SOL_MINT,
                swap_from_amount=500.0,
                swap_to_amount=5.0
            )
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        raw_price = 5.0 / 500.0  # 0.01
        expected_scaled_price = raw_price * feed.PRICE_SCALE_FACTOR  # 0.01 * 1e9 = 10,000,000
        actual_price = feed.df['transaction_price'].iloc[0]
        
        self.assertAlmostEqual(actual_price, expected_scaled_price, places=6)
    
    def test_non_sol_trade_returns_zero(self):
        """Test that non-SOL trades return zero price."""
        # Create test transaction: Token A → Token B (no SOL involved)
        transaction_data = pd.DataFrame([
            self.create_test_transaction(
                timestamp=self.base_time,
                swap_from_mint='TokenA123',
                swap_to_mint='TokenB456',
                swap_from_amount=100.0,
                swap_to_amount=200.0
            )
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        actual_price = feed.df['transaction_price'].iloc[0]
        self.assertEqual(actual_price, 0.0)
    
    def test_zero_amount_returns_zero(self):
        """Test that transactions with zero amounts return zero price."""
        # Test SOL → Token with zero token amount
        transaction_data = pd.DataFrame([
            self.create_test_transaction(
                timestamp=self.base_time,
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=10.0,
                swap_to_amount=0.0  # Zero token amount
            )
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        actual_price = feed.df['transaction_price'].iloc[0]
        self.assertEqual(actual_price, 0.0)
    
    def test_ohlcv_single_transaction(self):
        """Test OHLCV calculation with single transaction (scaled prices)."""
        transaction_data = pd.DataFrame([
            self.create_test_transaction(
                timestamp=self.base_time,
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=10.0,
                swap_to_amount=1000.0
            )
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        # Calculate OHLCV for a timestamp 30 seconds after the transaction
        sample_timestamp = self.base_time + timedelta(seconds=30)
        ohlcv = feed._calculate_ohlcv_from_transactions(sample_timestamp)
        
        raw_price = 0.01  # 10 / 1000
        expected_scaled_price = raw_price * feed.PRICE_SCALE_FACTOR  # 0.01 * 1e9 = 10,000,000
        expected_volume = 10.0  # SOL amount
        
        self.assertAlmostEqual(ohlcv['open'], expected_scaled_price, places=6)
        self.assertAlmostEqual(ohlcv['high'], expected_scaled_price, places=6)
        self.assertAlmostEqual(ohlcv['low'], expected_scaled_price, places=6)
        self.assertAlmostEqual(ohlcv['close'], expected_scaled_price, places=6)
        self.assertAlmostEqual(ohlcv['volume'], expected_volume, places=6)
    
    def test_ohlcv_multiple_transactions(self):
        """Test OHLCV calculation with multiple transactions (scaled prices)."""
        # Create multiple transactions with different prices
        base_time = self.base_time
        transaction_data = pd.DataFrame([
            # Transaction 1: price = 0.01 (10/1000)
            self.create_test_transaction(
                timestamp=base_time,
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=10.0,
                swap_to_amount=1000.0
            ),
            # Transaction 2: price = 0.02 (20/1000) 
            self.create_test_transaction(
                timestamp=base_time + timedelta(seconds=10),
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=20.0,
                swap_to_amount=1000.0
            ),
            # Transaction 3: price = 0.005 (5/1000)
            self.create_test_transaction(
                timestamp=base_time + timedelta(seconds=20),
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=5.0,
                swap_to_amount=1000.0
            ),
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        # Calculate OHLCV for a timestamp 30 seconds after first transaction
        sample_timestamp = base_time + timedelta(seconds=30)
        ohlcv = feed._calculate_ohlcv_from_transactions(sample_timestamp)
        
        # Expected values (scaled)
        expected_open = 0.01 * feed.PRICE_SCALE_FACTOR    # First transaction
        expected_high = 0.02 * feed.PRICE_SCALE_FACTOR    # Highest price
        expected_low = 0.005 * feed.PRICE_SCALE_FACTOR    # Lowest price  
        expected_close = 0.005 * feed.PRICE_SCALE_FACTOR  # Last transaction
        expected_volume = 35.0  # Total SOL: 10 + 20 + 5
        
        self.assertAlmostEqual(ohlcv['open'], expected_open, places=6)
        self.assertAlmostEqual(ohlcv['high'], expected_high, places=6)
        self.assertAlmostEqual(ohlcv['low'], expected_low, places=6)
        self.assertAlmostEqual(ohlcv['close'], expected_close, places=6)
        self.assertAlmostEqual(ohlcv['volume'], expected_volume, places=6)
    
    def test_ohlcv_empty_window_fallback(self):
        """Test OHLCV fallback when primary window is empty."""
        # Create transaction outside the primary 60s window
        old_transaction_time = self.base_time - timedelta(seconds=90)  # 90 seconds ago
        transaction_data = pd.DataFrame([
            self.create_test_transaction(
                timestamp=old_transaction_time,
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=10.0,
                swap_to_amount=1000.0
            )
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        # Sample timestamp with empty primary window (last 60s)
        sample_timestamp = self.base_time
        ohlcv = feed._calculate_ohlcv_from_transactions(sample_timestamp)
        
        # Should fallback to close price from -120s to -60s window (scaled)
        expected_fallback_price = 0.01 * feed.PRICE_SCALE_FACTOR  # (10 / 1000) * 1e9
        
        self.assertAlmostEqual(ohlcv['open'], expected_fallback_price, places=6)
        self.assertAlmostEqual(ohlcv['high'], expected_fallback_price, places=6)
        self.assertAlmostEqual(ohlcv['low'], expected_fallback_price, places=6)
        self.assertAlmostEqual(ohlcv['close'], expected_fallback_price, places=6)
        self.assertEqual(ohlcv['volume'], 0.0)  # No volume in primary window
    
    def test_ohlcv_ultimate_fallback(self):
        """Test OHLCV ultimate fallback when no transactions found."""
        # Create empty transaction data
        transaction_data = pd.DataFrame(columns=[
            'block_timestamp', 'swap_from_mint', 'swap_to_mint',
            'swap_from_amount', 'swap_to_amount', 'succeeded',
            'is_buy', 'sol_amount', 'swapper'
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        # Sample timestamp with no transactions anywhere
        sample_timestamp = self.base_time
        ohlcv = feed._calculate_ohlcv_from_transactions(sample_timestamp)
        
        # Should use ultimate fallback price of 1.0 * 1e9 (scaled)
        expected_fallback_price = 1.0 * feed.PRICE_SCALE_FACTOR
        
        self.assertAlmostEqual(ohlcv['open'], expected_fallback_price, places=6)
        self.assertAlmostEqual(ohlcv['high'], expected_fallback_price, places=6)
        self.assertAlmostEqual(ohlcv['low'], expected_fallback_price, places=6)
        self.assertAlmostEqual(ohlcv['close'], expected_fallback_price, places=6)
        self.assertEqual(ohlcv['volume'], 0.0)
    
    def test_price_calculation_statistics(self):
        """Test price calculation statistics and coverage."""
        # Create mixed transaction data
        transaction_data = pd.DataFrame([
            # Valid SOL trade
            self.create_test_transaction(
                timestamp=self.base_time,
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=10.0,
                swap_to_amount=1000.0
            ),
            # Non-SOL trade (should get 0 price)
            self.create_test_transaction(
                timestamp=self.base_time + timedelta(seconds=10),
                swap_from_mint='TokenA',
                swap_to_mint='TokenB',
                swap_from_amount=100.0,
                swap_to_amount=200.0
            ),
            # Invalid SOL trade (zero amount)
            self.create_test_transaction(
                timestamp=self.base_time + timedelta(seconds=20),
                swap_from_mint=self.SOL_MINT,
                swap_to_mint=self.TOKEN_MINT,
                swap_from_amount=5.0,
                swap_to_amount=0.0
            ),
        ])
        
        feed = SolanaTransactionFeed(transaction_data=transaction_data)
        
        # Check price coverage
        valid_prices = feed.df[feed.df['transaction_price'] > 0]
        total_transactions = len(feed.df)
        valid_count = len(valid_prices)
        
        self.assertEqual(total_transactions, 3)
        self.assertEqual(valid_count, 1)  # Only first transaction should have valid price
        
        # Check the valid price value (scaled)
        raw_price = 0.01  # 10 / 1000
        expected_scaled_price = raw_price * feed.PRICE_SCALE_FACTOR
        actual_price = valid_prices['transaction_price'].iloc[0]
        self.assertAlmostEqual(actual_price, expected_scaled_price, places=6)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)