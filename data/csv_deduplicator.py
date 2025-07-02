#!/usr/bin/env python3
"""
Simple CSV deduplicator - loops through CSV files in a folder and deduplicates them.
"""

import pandas as pd
import os
from pathlib import Path
import sys

def deduplicate_csv(input_file, output_file, dedup_columns=None):
    """
    Deduplicate a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output deduplicated CSV file  
        dedup_columns: List of columns to use for deduplication (None = all columns)
    """
    
    print(f"📄 Processing: {os.path.basename(input_file)}")
    
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        original_rows = len(df)
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        
        print(f"   📊 Input: {original_rows:,} rows, {file_size_mb:.1f} MB")
        
        # Filter out any __row_index columns from deduplication
        if dedup_columns:
            dedup_columns = [col for col in dedup_columns if col in df.columns and not col.startswith('__row_index')]
            if not dedup_columns:
                # If no valid columns remain, use all non-index columns
                dedup_columns = [col for col in df.columns if not col.startswith('__row_index')]
        else:
            # Use all columns except __row_index columns
            dedup_columns = [col for col in df.columns if not col.startswith('__row_index')]
        
        # Deduplicate
        print(f"   🎯 Dedup columns: {', '.join(dedup_columns)}")
        if dedup_columns:
            df_dedup = df.drop_duplicates(subset=dedup_columns)
            print(f"   🎯 Dedup columns: {', '.join(dedup_columns[:5])}{'...' if len(dedup_columns) > 5 else ''}")
        else:
            df_dedup = df.drop_duplicates()
            print(f"   🎯 Dedup columns: All columns")
        
        final_rows = len(df_dedup)
        duplicates_removed = original_rows - final_rows
        dedup_pct = (duplicates_removed / original_rows) * 100 if original_rows > 0 else 0
        
        # Save deduplicated file
        df_dedup.to_csv(output_file, index=False)
        
        output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"   ✅ Complete: {final_rows:,} unique rows saved")
        print(f"   🗑️  Removed: {duplicates_removed:,} duplicates ({dedup_pct:.1f}%)")
        print(f"   💾 Output size: {output_size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def process_folder(folder_path, output_folder=None, dedup_columns=None):
    """
    Process all CSV files in a folder.
    
    Args:
        folder_path: Input folder containing CSV files
        output_folder: Output folder (default: input_folder/deduped/)
        dedup_columns: List of columns to use for deduplication
    """
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return False
    
    # Find CSV files
    csv_files = list(folder.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith('.')]  # Skip hidden files
    
    if not csv_files:
        print(f"❌ No CSV files found in {folder_path}")
        return False
    
    # Set up output folder
    if output_folder:
        output_dir = Path(output_folder)
    else:
        output_dir = folder / "deduped"
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 CSV Deduplication")
    print(f"=" * 50)
    print(f"📁 Input folder: {folder}")
    print(f"📂 Output folder: {output_dir}")
    print(f"📄 Found {len(csv_files)} CSV files")
    if dedup_columns:
        print(f"🎯 Dedup columns: {', '.join(dedup_columns)}")
    else:
        print(f"🎯 Dedup columns: All columns")
    print()
    
    # Process each file
    successful = 0
    failed = 0
    total_original_rows = 0
    total_final_rows = 0
    
    for csv_file in csv_files:
        output_file = output_dir / f"{csv_file.stem}_deduped{csv_file.suffix}"
        
        success = deduplicate_csv(str(csv_file), str(output_file), dedup_columns)
        
        if success:
            successful += 1
            # Count rows for summary
            try:
                original_rows = len(pd.read_csv(csv_file, nrows=None))
                final_rows = len(pd.read_csv(output_file, nrows=None))
                total_original_rows += original_rows
                total_final_rows += final_rows
            except:
                pass
        else:
            failed += 1
        
        print()
    
    # Summary
    total_duplicates = total_original_rows - total_final_rows
    print(f"🎉 Processing Complete!")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Total original rows: {total_original_rows:,}")
    print(f"   📊 Total final rows: {total_final_rows:,}")
    print(f"   🗑️  Total duplicates removed: {total_duplicates:,}")
    if total_original_rows > 0:
        print(f"   📈 Overall dedup rate: {total_duplicates/total_original_rows*100:.1f}%")
    print(f"   📂 Output location: {output_dir}")
    
    return successful > 0

def main():
    """Main function."""
    # Default paths for Solana trading data
    default_input_folder = "/Volumes/Extreme SSD/trading_data/solana/first_day"
    default_output_folder = "/Volumes/Extreme SSD/trading_data/solana/first_day_dedup"
    
    if len(sys.argv) < 2:
        print("Usage: python csv_deduplicator.py [folder_path] [output_folder] [--columns col1 col2 ...]")
        print()
        print("Examples:")
        print("  python csv_deduplicator.py  # Uses default Solana data location")
        print(f"  python csv_deduplicator.py {default_input_folder}")
        print("  python csv_deduplicator.py data/solana/ data/clean/")
        print("  python csv_deduplicator.py data/solana/ --columns mint swapper block_timestamp")
        print()
        print(f"Default input:  {default_input_folder}")
        print(f"Default output: {default_output_folder}")
        print()
        
        # Use default paths when no arguments provided
        folder_path = default_input_folder
        output_folder = default_output_folder
        dedup_columns = None
    else:
        folder_path = sys.argv[1]
        output_folder = None
        dedup_columns = None
        
        # Parse arguments
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--columns":
                # Collect all remaining arguments as column names
                dedup_columns = args[i+1:]
                break
            else:
                # Assume it's the output folder
                output_folder = args[i]
            i += 1
        
        # Use default output folder if none specified
        if output_folder is None:
            output_folder = default_output_folder
    
    # Process the folder
    success = process_folder(folder_path, output_folder, dedup_columns)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 