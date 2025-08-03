#!/usr/bin/env python3
"""
Convert Transaction Data to OHLVC + Signals Format

This script implements the conversion plan to transform per-transaction data into
OHLVC candlestick data with pre-computed ML signals, making it compatible with
standard Backtrader workflows.

Process:
1. Pull feature table for a coin from DuckDB
2. Generate ML signals using classification model and regime clustering
3. Pull transaction data and create OHLVC candles using feature sample timestamps
4. Merge signals with OHLVC data
5. Export to CSV for backtesting

Based on: backtesting/data/convert_plan.md
"""

import sys
import os
import pandas as pd
import numpy as np
import duckdb
import joblib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add project root to path
sys.path.append('/Users/noel/projects/trading_eda')

from solana.inference.classification_forward.classification_inference import ClassificationInference


class OHLVCSignalConverter:
    """
    Convert transaction data to OHLVC format with pre-computed ML signals
    """
    
    def __init__(self, 
                 duckdb_path: str = "/Volumes/Extreme SSD/DuckDB/solana.duckdb",
                 model_dir: str = "/Volumes/Extreme SSD/trading_data/solana/models/classification_forward",
                 regime_model_path: str = "/Volumes/Extreme SSD/trading_data/solana/models/regime_clustering/regime_classifier.pkl",
                 scaler_path: str = "/Volumes/Extreme SSD/trading_data/solana/models/regime_clustering/feature_scaler.pkl",
                 output_dir: str = "/Users/noel/projects/trading_eda/backtesting/data/ohlvc_signals"):
        
        self.duckdb_path = duckdb_path
        self.model_dir = model_dir
        self.regime_model_path = regime_model_path
        self.scaler_path = scaler_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ml_inference = ClassificationInference(model_dir=model_dir)
        
        # Load regime models
        self.regime_classifier = joblib.load(regime_model_path)
        self.scaler = joblib.load(scaler_path)
        
        print(f"✅ Initialized OHLVCSignalConverter")
        print(f"   📊 Model info: {self.ml_inference.get_model_info()}")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def get_available_coins(self, limit: Optional[int] = None) -> List[str]:
        """Get list of available coins from the features table"""
        
        query = """
        SELECT DISTINCT coin_id 
        FROM classification_forward_features 
        ORDER BY coin_id
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with duckdb.connect(self.duckdb_path) as conn:
            result = conn.execute(query).fetchall()
            coins = [row[0] for row in result]
        
        print(f"📋 Found {len(coins)} coins in features table")
        return coins
    
    def load_features_for_coin(self, coin_id: str) -> pd.DataFrame:
        """Load classification features for a specific coin"""
        
        query = """
        SELECT * 
        FROM classification_forward_features 
        WHERE coin_id = ?
        ORDER BY sample_timestamp
        """
        
        with duckdb.connect(self.duckdb_path) as conn:
            features_df = conn.execute(query, [coin_id]).df()
        
        print(f"📊 Loaded {len(features_df)} feature records for coin {coin_id}")
        return features_df
    
    def generate_ml_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML signals using classification model and regime clustering"""
        
        print("🧠 Generating ML predictions...")
        
        # Prepare features for ML model (remove metadata columns)
        ml_features = features_df.drop(columns=[
            'coin_id', 'sample_timestamp', 'total_transactions', 
            'is_profitable_300s', 'forward_buy_volume_300s', 'forward_sell_volume_300s'
        ], errors='ignore')
        
        # Get ML predictions
        prediction_result = self.ml_inference.predict_full_output(
            ml_features, 
            threshold=0.1  # Low threshold to get all predictions
        )
        
        # Generate regime classifications
        print("🎪 Classifying market regimes...")
        X_scaled = self.scaler.transform(ml_features.values)
        regimes = self.regime_classifier.predict(X_scaled)
        
        # Create signals dataframe
        signals_df = pd.DataFrame({
            'sample_timestamp': features_df['sample_timestamp'],
            'coin_id': features_df['coin_id'],
            'ml_prediction_score': prediction_result['scores'],
            'ml_prediction_label': prediction_result['labels'],
            'ml_confidence': prediction_result['confidence'],
            'regime': regimes,
        })
        
        # Generate trading signals based on regime anomaly strategy
        signals_df['safe_long_signal'] = (signals_df['ml_prediction_score'] >= 0.65).astype(int)
        
        # Regime 1 contrarian signals (prob ranges: 0.13-0.14, 0.16-0.17)
        regime_1_mask = signals_df['regime'] == 1
        contrarian_range_1 = (signals_df['ml_prediction_score'] >= 0.13) & (signals_df['ml_prediction_score'] <= 0.14)
        contrarian_range_2 = (signals_df['ml_prediction_score'] >= 0.16) & (signals_df['ml_prediction_score'] <= 0.17)
        regime_1_contrarian = regime_1_mask & (contrarian_range_1 | contrarian_range_2)
        
        signals_df['regime_1_contrarian_signal'] = regime_1_contrarian.astype(int)
        
        # Signal type classification
        signals_df['signal_type'] = 'none'
        signals_df.loc[signals_df['safe_long_signal'] == 1, 'signal_type'] = 'safe_long'
        signals_df.loc[signals_df['regime_1_contrarian_signal'] == 1, 'signal_type'] = 'regime_1_contrarian'
        
        # Add time since start (will be calculated relative to first timestamp)
        first_timestamp = signals_df['sample_timestamp'].min()
        signals_df['hours_since_start'] = (
            pd.to_datetime(signals_df['sample_timestamp']) - pd.to_datetime(first_timestamp)
        ).dt.total_seconds() / 3600
        
        print(f"📈 Generated signals summary:")
        print(f"   Safe long signals: {signals_df['safe_long_signal'].sum()}")
        print(f"   Regime 1 contrarian signals: {signals_df['regime_1_contrarian_signal'].sum()}")
        print(f"   Regime distribution: {signals_df['regime'].value_counts().to_dict()}")
        
        return signals_df
    
    def load_transactions_for_coin(self, coin_id: str) -> pd.DataFrame:
        """Load transaction data for a specific coin"""
        
        query = f"""
        SELECT 
            block_timestamp,
            mint,
            swapper,
            succeeded,
            CASE 
                WHEN swap_from_mint = 'So11111111111111111111111111111111111111112' THEN 1 
                ELSE 0 
            END as is_buy,
            CASE 
                WHEN swap_from_mint = 'So11111111111111111111111111111111111111112' 
                    AND swap_to_amount > 0 
                THEN swap_from_amount / swap_to_amount 
                WHEN swap_to_mint = 'So11111111111111111111111111111111111111112'
                    AND swap_from_amount > 0
                THEN swap_to_amount / swap_from_amount 
                ELSE 0.0
            END as price_per_token,
            CASE 
                WHEN swap_from_mint = 'So11111111111111111111111111111111111111112' 
                THEN swap_from_amount 
                ELSE swap_to_amount 
            END as sol_amount,
            swap_from_amount,
            swap_to_amount,
            swap_from_mint,
            swap_to_mint
        FROM first_day_trades
        WHERE mint = ?
        AND succeeded = TRUE
        AND (swap_from_mint = 'So11111111111111111111111111111111111111112' 
             OR swap_to_mint = 'So11111111111111111111111111111111111111112')
        AND mint != 'So11111111111111111111111111111111111111112'
        AND swap_from_amount > 0 
        AND swap_to_amount > 0
        ORDER BY block_timestamp
        """
        
        with duckdb.connect(self.duckdb_path) as conn:
            transactions_df = conn.execute(query, [coin_id]).df()
        
        print(f"💱 Loaded {len(transactions_df)} transactions for coin {coin_id}")
        return transactions_df
    
    def create_ohlvc_candles(self, transactions_df: pd.DataFrame, 
                           sample_timestamps: pd.Series, 
                           candle_interval_minutes: int = 1) -> pd.DataFrame:
        """
        Create OHLVC candles using exact sample timestamps from features table
        """
        
        print(f"📊 Creating {candle_interval_minutes}-minute OHLVC candles...")
        
        # Ensure timestamp columns are datetime
        transactions_df['block_timestamp'] = pd.to_datetime(transactions_df['block_timestamp'])
        sample_timestamps = pd.to_datetime(sample_timestamps)
        
        candles = []
        
        for sample_time in sample_timestamps:
            # Define candle window (1 minute candle ending at sample_time)
            candle_start = sample_time - timedelta(minutes=candle_interval_minutes)
            candle_end = sample_time
            
            # Filter transactions in this window
            window_mask = (
                (transactions_df['block_timestamp'] > candle_start) & 
                (transactions_df['block_timestamp'] <= candle_end)
            )
            window_txns = transactions_df[window_mask]
            
            if len(window_txns) == 0:
                # No transactions in this window - use previous close or 0
                if len(candles) > 0:
                    last_close = candles[-1]['close']  # Already scaled from previous candle
                    candles.append({
                        'timestamp': sample_time,
                        'open': last_close,
                        'high': last_close,
                        'low': last_close,
                        'close': last_close,
                        'volume': 0,
                        'transaction_count': 0
                    })
                else:
                    # First candle with no data
                    candles.append({
                        'timestamp': sample_time,
                        'open': 0,
                        'high': 0,
                        'low': 0,
                        'close': 0,
                        'volume': 0,
                        'transaction_count': 0
                    })
            else:
                # Calculate OHLVC from transactions
                prices = window_txns['price_per_token'] * 1e9  # Scale factor to make prices more readable
                volumes = window_txns['sol_amount']
                
                candles.append({
                    'timestamp': sample_time,
                    'open': prices.iloc[0],
                    'high': prices.max(),
                    'low': prices.min(),
                    'close': prices.iloc[-1],
                    'volume': volumes.sum(),
                    'transaction_count': len(window_txns)
                })
        
        candles_df = pd.DataFrame(candles)
        
        # Handle any remaining NaN values
        candles_df = candles_df.fillna(method='ffill').fillna(0)
        
        print(f"📊 Created {len(candles_df)} candles")
        print(f"   Price range: ${candles_df['close'].min():.4f} - ${candles_df['close'].max():.4f}")
        print(f"   Volume range: {candles_df['volume'].min():.2f} - {candles_df['volume'].max():.2f}")
        print(f"   Avg transactions per candle: {candles_df['transaction_count'].mean():.1f}")
        
        return candles_df
    
    def merge_ohlvc_with_signals(self, candles_df: pd.DataFrame, 
                               signals_df: pd.DataFrame) -> pd.DataFrame:
        """Merge OHLVC candles with ML signals"""
        
        print("🔗 Merging OHLVC data with signals...")
        
        # Ensure timestamp alignment
        candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'])
        signals_df['sample_timestamp'] = pd.to_datetime(signals_df['sample_timestamp'])
        
        # Merge on timestamp
        merged_df = pd.merge(
            candles_df,
            signals_df,
            left_on='timestamp',
            right_on='sample_timestamp',
            how='inner'
        )
        
        # Rename timestamp to datetime for Backtrader compatibility
        merged_df = merged_df.rename(columns={'timestamp': 'datetime'})
        
        # Add openinterest column (required by Backtrader, set to 0 for crypto)
        merged_df['openinterest'] = 0
        
        # Select and order final columns (Backtrader standard + signals)
        final_columns = [
            'datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
            'ml_prediction_score', 'ml_prediction_label', 'ml_confidence', 'regime',
            'safe_long_signal', 'regime_1_contrarian_signal', 'signal_type',
            'hours_since_start', 'transaction_count', 'coin_id'
        ]
        
        final_df = merged_df[final_columns].copy()
        
        print(f"🔗 Merged dataset: {len(final_df)} records")
        print(f"   Signal distribution: {final_df['signal_type'].value_counts().to_dict()}")
        
        return final_df
    
    def convert_coin_to_ohlvc_signals(self, coin_id: str, 
                                    candle_interval_minutes: int = 1) -> pd.DataFrame:
        """Complete conversion pipeline for a single coin"""
        
        print(f"\n🪙 Converting coin: {coin_id}")
        print("=" * 60)
        
        # Step 1: Load features and generate signals
        features_df = self.load_features_for_coin(coin_id)
        if len(features_df) == 0:
            print(f"⚠️ No features found for coin {coin_id}")
            return pd.DataFrame()
        
        signals_df = self.generate_ml_signals(features_df)
        
        # Step 2: Load transactions and create OHLVC
        transactions_df = self.load_transactions_for_coin(coin_id)
        if len(transactions_df) == 0:
            print(f"⚠️ No transactions found for coin {coin_id}")
            return pd.DataFrame()
        
        candles_df = self.create_ohlvc_candles(
            transactions_df, 
            signals_df['sample_timestamp'],
            candle_interval_minutes
        )
        
        # Step 3: Merge OHLVC with signals
        final_df = self.merge_ohlvc_with_signals(candles_df, signals_df)
        
        # Step 4: Add trading window validation
        final_df['valid_trading_window'] = (
            (final_df['hours_since_start'] >= 1) & 
            (final_df['hours_since_start'] <= 10)
        ).astype(int)
        
        return final_df
    
    def save_to_csv(self, df: pd.DataFrame, coin_id: str) -> str:
        """Save dataframe to CSV file"""
        
        output_file = self.output_dir / f"{coin_id}_ohlvc_signals.csv"
        df.to_csv(output_file, index=False)
        
        print(f"💾 Saved to: {output_file}")
        return str(output_file)
    
    def convert_multiple_coins(self, coin_ids: List[str], 
                             candle_interval_minutes: int = 1) -> Dict[str, str]:
        """Convert multiple coins to OHLVC + signals format"""
        
        results = {}
        
        for i, coin_id in enumerate(coin_ids, 1):
            print(f"\n🚀 Processing coin {i}/{len(coin_ids)}: {coin_id}")
            
            try:
                # Convert coin data
                final_df = self.convert_coin_to_ohlvc_signals(coin_id, candle_interval_minutes)
                
                if len(final_df) > 0:
                    # Save to CSV
                    output_file = self.save_to_csv(final_df, coin_id)
                    results[coin_id] = output_file
                    
                    print(f"✅ Successfully converted {coin_id}: {len(final_df)} records")
                else:
                    print(f"❌ Failed to convert {coin_id}: No data generated")
                    results[coin_id] = None
                    
            except Exception as e:
                print(f"❌ Error processing {coin_id}: {e}")
                results[coin_id] = None
        
        return results
    
    def create_summary_report(self, results: Dict[str, str]):
        """Create a summary report of the conversion process"""
        
        successful = sum(1 for v in results.values() if v is not None)
        failed = len(results) - successful
        
        print(f"\n📋 CONVERSION SUMMARY")
        print("=" * 50)
        print(f"Total coins processed: {len(results)}")
        print(f"Successfully converted: {successful}")
        print(f"Failed conversions: {failed}")
        
        if successful > 0:
            print(f"\n✅ Successful conversions:")
            for coin_id, output_file in results.items():
                if output_file:
                    print(f"   {coin_id} → {output_file}")
        
        if failed > 0:
            print(f"\n❌ Failed conversions:")
            for coin_id, output_file in results.items():
                if output_file is None:
                    print(f"   {coin_id}")


def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description='Convert transaction data to OHLVC + signals format')
    parser.add_argument('--coins', nargs='+', help='Specific coin IDs to convert')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of coins to process (default: 10)')
    parser.add_argument('--candle-interval', type=int, default=1, help='Candle interval in minutes (default: 1)')
    parser.add_argument('--all', action='store_true', help='Process all available coins')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = OHLVCSignalConverter()
    
    # Determine which coins to process
    if args.coins:
        coin_ids = args.coins
        print(f"🎯 Processing specified coins: {coin_ids}")
    elif args.all:
        coin_ids = converter.get_available_coins()
        print(f"🌍 Processing all available coins: {len(coin_ids)} coins")
    else:
        coin_ids = converter.get_available_coins(limit=args.limit)
        print(f"📊 Processing first {len(coin_ids)} coins")
    
    # Convert coins
    results = converter.convert_multiple_coins(coin_ids, args.candle_interval)
    
    # Generate summary
    converter.create_summary_report(results)
    
    print(f"\n🎉 Conversion process completed!")
    print(f"📁 Output directory: {converter.output_dir}")


if __name__ == "__main__":
    main()