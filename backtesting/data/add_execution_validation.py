#!/usr/bin/env python3
"""
Add Execution Validation Signals to OHLVC Signal Files

This script implements the additional execution validation logic outlined in convert_plan.md:
1. Available to execute buy signal column (binary)
2. Coin size column (numerical) 
3. Available to execute sell signal column (binary)

The validation prevents execution errors by pre-checking market conditions and liquidity
constraints at the data level rather than strategy level.

Based on: backtesting/data/convert_plan.md (Additional signals section)
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class ExecutionValidationProcessor:
    """
    Add execution validation signals to existing OHLVC signal files
    """
    
    def __init__(self, 
                 input_dir: str = "/Users/noel/projects/trading_eda/backtesting/data/ohlvc_signals",
                 output_dir: str = "/Users/noel/projects/trading_eda/backtesting/data/ohlvc_signals_validated",
                 fixed_sol_amount: float = 10.0,
                 holding_period: int = 3):
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_sol_amount = fixed_sol_amount  # Fixed SOL amount for each buy order
        self.holding_period = holding_period  # Number of bars to hold before sell
        
        print(f"✅ Initialized ExecutionValidationProcessor")
        print(f"   📂 Input directory: {self.input_dir}")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   💰 Fixed SOL amount per trade: {self.fixed_sol_amount}")
        print(f"   ⏱️ Holding period: {self.holding_period} bars")
    
    def get_available_csv_files(self) -> List[Path]:
        """Get list of available CSV files to process"""
        
        csv_files = list(self.input_dir.glob("*_ohlvc_signals.csv"))
        print(f"📋 Found {len(csv_files)} CSV files to process")
        return csv_files
    
    def load_ohlvc_signals(self, csv_file: Path) -> pd.DataFrame:
        """Load existing OHLVC signals CSV file"""
        
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded {len(df)} records from {csv_file.name}")
        
        # Ensure datetime column is properly parsed
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def add_execution_validation_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add three execution validation columns based on convert_plan.md logic:
        
        1. can_execute_buy: Check if next row has enough volume and favorable price
        2. coin_size: Track position size based on fixed SOL amount
        3. can_execute_sell: Check if sell can be executed with sufficient volume
        """
        
        print("🔍 Adding execution validation signals...")
        
        # Initialize new columns
        df = df.copy()
        df['can_execute_buy'] = 0
        df['coin_size'] = 0.0
        df['can_execute_sell'] = 0
        
        # Get buy signals (any signal that could trigger a buy)
        buy_signal_mask = (
            (df['safe_long_signal'] == 1) | 
            (df['regime_1_contrarian_signal'] == 1)
        )
        buy_signal_indices = df[buy_signal_mask].index.tolist()
        
        print(f"   📈 Found {len(buy_signal_indices)} potential buy signals")
        
        current_position_size = 0.0  # Track current coin holdings
        buy_row_index = None  # Track when the current position was bought
        sell_execution_row = None  # Track where the sell will actually execute
        
        for i in range(len(df)):
            # Check buy execution availability (only if no current position)
            if i in buy_signal_indices and current_position_size == 0.0:
                can_buy = self._check_buy_execution_availability(df, i)
                df.at[i, 'can_execute_buy'] = int(can_buy)
                
                if can_buy:
                    # Calculate coin size based on next row's open price
                    if i + 1 < len(df):
                        next_open_price = df.at[i + 1, 'open']
                        if next_open_price > 0:
                            current_position_size = self.fixed_sol_amount / next_open_price
                            buy_row_index = i
                            print(f"   💰 Buy executed at row {i}: {current_position_size:.6f} coins at price {next_open_price:.4f}")
            
            # Check sell execution availability (only after holding period)
            elif current_position_size > 0.0 and buy_row_index is not None and sell_execution_row is None:
                bars_held = i - buy_row_index
                if bars_held >= self.holding_period:
                    # Find where the sell can actually be executed
                    execution_row = self.find_sell_execution_row(df, i, current_position_size)
                    
                    if execution_row is not None:
                        # Mark the row BEFORE execution as sellable (signal row)
                        signal_row = execution_row - 1
                        if signal_row >= i:  # Make sure signal row is not before current eligibility
                            df.at[signal_row, 'can_execute_sell'] = 1
                            sell_execution_row = execution_row
                            execution_price = df.at[execution_row, 'open']
                            print(f"   💸 Sell signal at row {signal_row}, execution at row {execution_row}: {current_position_size:.6f} coins at price {execution_price:.4f} (held for {execution_row - buy_row_index} bars)")
            
            # Reset position if we've reached the sell execution row
            if sell_execution_row is not None and i == sell_execution_row:
                current_position_size = 0.0
                buy_row_index = None
                sell_execution_row = None
            
            # Update coin_size for current row
            df.at[i, 'coin_size'] = current_position_size
        
        buy_executed = df['can_execute_buy'].sum()
        sell_executed = df['can_execute_sell'].sum()
        
        print(f"   ✅ Validation results:")
        print(f"      Buy orders executable: {buy_executed}")
        print(f"      Sell orders executable: {sell_executed}")
        print(f"      Max position size: {df['coin_size'].max():.6f} coins")
        
        return df
    
    def _check_buy_execution_availability(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        Check if buy order can be executed based on next row conditions:
        1. Next row has enough volume (SOL) to execute the buy order
        2. Next row has a low price than current open price (favorable entry)
        """
        
        if current_idx + 1 >= len(df):
            return False
        
        current_row = df.iloc[current_idx]
        next_row = df.iloc[current_idx + 1]
        
        # Check 1: Sufficient volume in next row
        next_volume = next_row['volume']
        if next_volume < self.fixed_sol_amount:
            return False
        
        # Check 2: Favorable price (next low is lower than next open)
        next_open = next_row['open']
        next_low = next_row['low']
        
        if next_low >= next_open:  # Not favorable if low doesn't go below open
            return False
        
        return True
    
    def find_sell_execution_row(self, df: pd.DataFrame, start_idx: int, position_size: float) -> Optional[int]:
        """
        Find the first row with sufficient volume to execute the sell order
        Returns the row index where execution can actually happen, or None if not found
        """
        max_lookahead = min(10, len(df) - start_idx)
        
        for offset in range(1, max_lookahead + 1):
            check_idx = start_idx + offset
            if check_idx >= len(df):
                break
                
            check_volume = df.at[check_idx, 'volume']
            check_price = df.at[check_idx, 'open']
            required_volume = position_size * check_price
            
            if check_volume >= required_volume:
                return check_idx  # Return the actual execution row
                
        return None  # No adequate volume found
    
    def process_single_file(self, csv_file: Path) -> Optional[Path]:
        """Process a single CSV file and add execution validation signals"""
        
        print(f"\n🔄 Processing: {csv_file.name}")
        print("=" * 60)
        
        try:
            # Load existing data
            df = self.load_ohlvc_signals(csv_file)
            
            # Add validation signals
            validated_df = self.add_execution_validation_signals(df)
            
            # Save to output directory
            output_file = self.output_dir / csv_file.name.replace('_ohlvc_signals.csv', '_validated.csv')
            validated_df.to_csv(output_file, index=False)
            
            print(f"💾 Saved validated data to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")
            return None
    
    def process_multiple_files(self, csv_files: List[Path]) -> Dict[str, Optional[Path]]:
        """Process multiple CSV files"""
        
        results = {}
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n🚀 Processing file {i}/{len(csv_files)}")
            
            output_file = self.process_single_file(csv_file)
            results[csv_file.name] = output_file
        
        return results
    
    def create_summary_report(self, results: Dict[str, Optional[Path]]):
        """Create a summary report of the processing results"""
        
        successful = sum(1 for v in results.values() if v is not None)
        failed = len(results) - successful
        
        print(f"\n📋 PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(results)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed processing: {failed}")
        
        if successful > 0:
            print(f"\n✅ Successfully processed files:")
            for input_file, output_file in results.items():
                if output_file:
                    print(f"   {input_file} → {output_file.name}")
        
        if failed > 0:
            print(f"\n❌ Failed processing:")
            for input_file, output_file in results.items():
                if output_file is None:
                    print(f"   {input_file}")


def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(
        description='Add execution validation signals to OHLVC signal files'
    )
    parser.add_argument(
        '--input-dir', 
        default='/Users/noel/projects/trading_eda/backtesting/data/ohlvc_signals',
        help='Input directory containing OHLVC signal CSV files'
    )
    parser.add_argument(
        '--output-dir',
        default='/Users/noel/projects/trading_eda/backtesting/data/ohlvc_signals_validated', 
        help='Output directory for validated CSV files'
    )
    parser.add_argument(
        '--sol-amount',
        type=float,
        default=10.0,
        help='Fixed SOL amount per trade (default: 10.0)'
    )
    parser.add_argument(
        '--holding-period',
        type=int,
        default=3,
        help='Number of bars to hold position before selling (default: 3)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Specific CSV files to process (just filenames, not full paths)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of files to process'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ExecutionValidationProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fixed_sol_amount=args.sol_amount,
        holding_period=args.holding_period
    )
    
    # Determine which files to process
    if args.files:
        # Process specific files
        input_dir = Path(args.input_dir)
        csv_files = [input_dir / filename for filename in args.files]
        csv_files = [f for f in csv_files if f.exists()]
        print(f"🎯 Processing specified files: {len(csv_files)} files")
    else:
        # Process all files in directory
        csv_files = processor.get_available_csv_files()
        if args.limit:
            csv_files = csv_files[:args.limit]
            print(f"📊 Processing first {len(csv_files)} files")
    
    if not csv_files:
        print("⚠️ No CSV files found to process")
        return
    
    # Process files
    results = processor.process_multiple_files(csv_files)
    
    # Generate summary
    processor.create_summary_report(results)
    
    print(f"\n🎉 Processing completed!")
    print(f"📁 Output directory: {processor.output_dir}")


if __name__ == "__main__":
    main()