#!/usr/bin/env python3
"""
Data Utilities for Backtesting
Data loading, preprocessing, and signal generation utilities
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add project root to path
sys.path.append('/Users/noel/projects/trading_eda')


def load_ohlcv_data(data_file: str, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load OHLCV data from CSV file
    
    Args:
        data_file: Path to OHLCV data file
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data and datetime index
    """
    print(f"Loading data from: {data_file}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Handle datetime column
    datetime_col = None
    for col in ['timestamp', 'date', 'datetime']:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col:
        df['datetime'] = pd.to_datetime(df[datetime_col])
    else:
        print("Warning: No timestamp/date column found, creating synthetic dates")
        df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    
    df.set_index('datetime', inplace=True)
    
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        # Try common alternative column names
        column_mapping = {
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        
        for alt_col, std_col in column_mapping.items():
            if alt_col in df.columns and std_col in missing_cols:
                df[std_col] = df[alt_col]
                print(f"  Mapped {alt_col} -> {std_col}")
    
    # Filter by date range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def add_ml_signals(df: pd.DataFrame, 
                   signal_source: str = 'inference',
                   model_path: Optional[str] = None,
                   **kwargs) -> pd.DataFrame:
    """
    Add ML signals to OHLCV data
    
    Args:
        df: OHLCV dataframe
        signal_source: Source of signals ('inference', 'random', 'file')
        model_path: Path to ML model (for inference)
        **kwargs: Additional parameters
    
    Returns:
        DataFrame with ML signals added
    """
    print(f"Adding ML signals using source: {signal_source}")
    
    if signal_source == 'inference':
        return _add_inference_signals(df, model_path, **kwargs)
    elif signal_source == 'random':
        return _add_random_signals(df, **kwargs)
    elif signal_source == 'file':
        return _add_file_signals(df, **kwargs)
    else:
        raise ValueError(f"Unknown signal source: {signal_source}")


def _add_inference_signals(df: pd.DataFrame, 
                          model_path: Optional[str] = None,
                          **kwargs) -> pd.DataFrame:
    """
    Add signals using ML inference
    """
    try:
        if model_path:
            # Load specific model
            import joblib
            model = joblib.load(model_path)
            print(f"Loaded model from: {model_path}")
            
            # Basic prediction (adapt based on your model interface)
            if hasattr(model, 'predict_proba'):
                # Prepare features (simplified - adapt to your model's needs)
                features = df[['open', 'high', 'low', 'close', 'volume']].fillna(0)
                probabilities = model.predict_proba(features)
                df['ml_signal'] = probabilities[:, 1]  # Probability of positive class
                df['ml_confidence'] = np.max(probabilities, axis=1)
            else:
                print("Model does not support probability prediction")
                return _add_random_signals(df, **kwargs)
        else:
            # Use default inference engine
            from solana.inference.classification_forward.classification_inference import ClassificationInference
            inference = ClassificationInference()
            
            results = inference.predict_full_output(df)
            df['ml_signal'] = results['scores']
            df['ml_confidence'] = results['confidence']
        
        print(f"✅ ML signals generated successfully")
        print(f"   Average signal: {df['ml_signal'].mean():.3f}")
        print(f"   Average confidence: {df['ml_confidence'].mean():.3f}")
        
    except Exception as e:
        print(f"⚠️  Could not generate ML signals: {e}")
        print("   Falling back to random signals")
        df = _add_random_signals(df, **kwargs)
    
    return df


def _add_random_signals(df: pd.DataFrame, 
                       seed: int = 42,
                       signal_range: tuple = (0.3, 0.8),
                       confidence_range: tuple = (0.5, 0.9),
                       **kwargs) -> pd.DataFrame:
    """
    Add random signals for testing
    """
    np.random.seed(seed)
    n_samples = len(df)
    
    df['ml_signal'] = np.random.uniform(signal_range[0], signal_range[1], n_samples)
    df['ml_confidence'] = np.random.uniform(confidence_range[0], confidence_range[1], n_samples)
    
    print(f"Generated random signals for {n_samples} samples")
    return df


def _add_file_signals(df: pd.DataFrame, 
                     signal_file: str,
                     signal_col: str = 'signal',
                     confidence_col: str = 'confidence',
                     **kwargs) -> pd.DataFrame:
    """
    Add signals from external file
    """
    print(f"Loading signals from: {signal_file}")
    
    if not os.path.exists(signal_file):
        raise FileNotFoundError(f"Signal file not found: {signal_file}")
    
    signal_df = pd.read_csv(signal_file)
    
    # Try to match by index or merge by datetime
    if len(signal_df) == len(df):
        df['ml_signal'] = signal_df[signal_col].values
        df['ml_confidence'] = signal_df[confidence_col].values if confidence_col in signal_df.columns else 0.7
    else:
        print("Warning: Signal file length doesn't match data length")
        # Try to merge by datetime if available
        if 'datetime' in signal_df.columns:
            signal_df['datetime'] = pd.to_datetime(signal_df['datetime'])
            signal_df.set_index('datetime', inplace=True)
            
            df = df.join(signal_df[[signal_col, confidence_col]], how='left')
            df['ml_signal'] = df[signal_col].fillna(0.5)
            df['ml_confidence'] = df[confidence_col].fillna(0.5)
        else:
            raise ValueError("Cannot align signal data with OHLCV data")
    
    print(f"Signals loaded successfully")
    return df


def prepare_data_for_backtest(data_file: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              signal_source: str = 'inference',
                              **signal_kwargs) -> pd.DataFrame:
    """
    Complete data preparation pipeline for backtesting
    
    Args:
        data_file: Path to OHLCV data file
        start_date: Start date for backtesting (YYYY-MM-DD)
        end_date: End date for backtesting (YYYY-MM-DD)
        signal_source: Source of ML signals ('inference', 'random', 'file')
        **signal_kwargs: Additional parameters for signal generation
    
    Returns:
        DataFrame ready for backtesting
    """
    print("=== PREPARING DATA FOR BACKTEST ===")
    
    # Load OHLCV data
    df = load_ohlcv_data(data_file, start_date, end_date)
    
    # Add ML signals
    df = add_ml_signals(df, signal_source, **signal_kwargs)
    
    # Add technical indicators (optional)
    df = add_technical_indicators(df)
    
    # Validate final data
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'ml_signal', 'ml_confidence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns after preparation: {missing_cols}")
    
    # Clean data
    df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"Data preparation complete: {len(df)} samples ready for backtesting")
    print(f"Signal range: [{df['ml_signal'].min():.3f}, {df['ml_signal'].max():.3f}]")
    print(f"Confidence range: [{df['ml_confidence'].min():.3f}, {df['ml_confidence'].max():.3f}]")
    
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to the dataframe
    
    Args:
        df: OHLCV dataframe
        
    Returns:
        DataFrame with technical indicators added
    """
    try:
        # Simple moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        print("Technical indicators added successfully")
        
    except Exception as e:
        print(f"Warning: Could not add technical indicators: {e}")
    
    return df


def create_sample_data(n_days: int = 365,
                      start_price: float = 100.0,
                      volatility: float = 0.02,
                      trend: float = 0.0005,
                      seed: int = 42) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing
    
    Args:
        n_days: Number of days to generate
        start_price: Starting price
        volatility: Daily volatility
        trend: Daily trend (drift)
        seed: Random seed
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    print(f"Creating sample data: {n_days} days")
    
    np.random.seed(seed)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate price series with trend and volatility
    returns = np.random.normal(trend, volatility, n_days)
    prices = [start_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 0.01))  # Ensure positive prices
    
    # Create OHLCV data
    df = pd.DataFrame({
        'datetime': dates,
        'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_days)
    })
    
    # Ensure OHLC relationships are correct
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    df.set_index('datetime', inplace=True)
    
    print(f"Sample data created: {len(df)} days, price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    return df


def validate_backtest_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data for backtesting
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        validation['valid'] = False
        validation['errors'].append(f"Missing required columns: {missing_cols}")
    
    # Check for missing values in critical columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns and df[col].isnull().any():
            validation['warnings'].append(f"Missing values in {col}")
    
    # Check OHLC relationships
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
        
        if invalid_high > 0:
            validation['warnings'].append(f"{invalid_high} rows with invalid high prices")
        if invalid_low > 0:
            validation['warnings'].append(f"{invalid_low} rows with invalid low prices")
    
    # Check for negative prices or volumes
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns and (df[col] <= 0).any():
            validation['warnings'].append(f"Non-positive values in {col}")
    
    # Generate stats
    validation['stats'] = {
        'n_samples': len(df),
        'date_range': f"{df.index.min()} to {df.index.max()}",
        'columns': list(df.columns),
        'has_signals': 'ml_signal' in df.columns
    }
    
    return validation
