#!/usr/bin/env python3
"""
Signal Generators for Trading Strategies
Pluggable components for generating trading signals from different sources
"""

import pandas as pd
import numpy as np
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

# Add project root to path
sys.path.append('/Users/noel/projects/trading_eda')


class SignalGenerator(ABC):
    """
    Abstract base class for signal generators
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Generate trading signal from current market data
        
        Args:
            data: Dictionary containing market data (OHLCV, indicators, etc.)
            
        Returns:
            Tuple of (signal_strength, confidence)
        """
        pass
    
    @abstractmethod
    def initialize(self, strategy_context: Any):
        """
        Initialize the signal generator with strategy context
        
        Args:
            strategy_context: Backtrader strategy instance
        """
        pass


class MLClassificationSignalGenerator(SignalGenerator):
    """
    Signal generator using ML classification models
    """
    
    def __init__(self, model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.ml_model = None
        self.feature_calculators = {}
    
    def initialize(self, strategy_context: Any):
        """Initialize ML model and feature calculators"""
        self.strategy_context = strategy_context
        
        # Try to load ML model
        try:
            if self.model_path:
                # Load specific model
                import joblib
                self.ml_model = joblib.load(self.model_path)
            else:
                # Use default inference engine
                from solana.inference.classification_forward.classification_inference import ClassificationInference
                self.ml_model = ClassificationInference()
            
            print("✅ ML classification model loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load ML model: {e}")
            self.ml_model = None
        
        # Initialize technical indicators for feature calculation
        import backtrader as bt
        self.feature_calculators = {
            'sma_short': bt.indicators.SimpleMovingAverage(strategy_context.datas[0], period=10),
            'sma_long': bt.indicators.SimpleMovingAverage(strategy_context.datas[0], period=30),
            'rsi': bt.indicators.RelativeStrengthIndex(strategy_context.datas[0], period=14),
            'macd': bt.indicators.MACD(strategy_context.datas[0])
        }
    
    def generate_signal(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """Generate signal using ML model"""
        if self.ml_model is None:
            return 0.5, 0.5  # Neutral signal with low confidence
        
        try:
            # First try to use pre-computed signals if available
            if 'ml_signal' in data and 'ml_confidence' in data:
                return data['ml_signal'], data['ml_confidence']
            
            # Otherwise, calculate features and predict
            features = self._calculate_features(data)
            if features is None:
                return 0.5, 0.5
            
            # Create feature dataframe
            feature_df = pd.DataFrame([features])
            
            # Get prediction
            if hasattr(self.ml_model, 'predict_full_output'):
                results = self.ml_model.predict_full_output(feature_df)
                signal = results['scores'][0]
                confidence = results['confidence'][0]
            else:
                # Fallback for basic sklearn models
                if hasattr(self.ml_model, 'predict_proba'):
                    proba = self.ml_model.predict_proba(feature_df)
                    signal = proba[0, 1]  # Probability of positive class
                    confidence = np.max(proba[0])
                else:
                    signal = self.ml_model.predict(feature_df)[0]
                    confidence = 0.7  # Default confidence
            
            return signal, confidence
            
        except Exception as e:
            print(f"Error in ML signal generation: {e}")
            return 0.5, 0.5
    
    def _calculate_features(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Calculate features for ML prediction"""
        try:
            # Basic price features
            features = {
                'close_price': data.get('close', 0),
                'open_price': data.get('open', 0),
                'high_price': data.get('high', 0),
                'low_price': data.get('low', 0),
                'volume': data.get('volume', 0),
            }
            
            # Technical indicators (if available)
            if self.feature_calculators:
                features.update({
                    'sma_10': self.feature_calculators['sma_short'][0] if not np.isnan(self.feature_calculators['sma_short'][0]) else 0,
                    'sma_30': self.feature_calculators['sma_long'][0] if not np.isnan(self.feature_calculators['sma_long'][0]) else 0,
                    'rsi_14': self.feature_calculators['rsi'][0] if not np.isnan(self.feature_calculators['rsi'][0]) else 50,
                    'macd': self.feature_calculators['macd'].macd[0] if not np.isnan(self.feature_calculators['macd'].macd[0]) else 0,
                    'macd_signal': self.feature_calculators['macd'].signal[0] if not np.isnan(self.feature_calculators['macd'].signal[0]) else 0,
                })
            
            # Price-based features
            if 'prev_close' in data:
                features['price_change'] = (data['close'] - data['prev_close']) / data['prev_close']
            
            if features['sma_30'] != 0:
                features['sma_ratio'] = features['sma_10'] / features['sma_30']
            else:
                features['sma_ratio'] = 1
            
            return features
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None


class TechnicalIndicatorSignalGenerator(SignalGenerator):
    """
    Signal generator based on technical indicators
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indicators = {}
    
    def initialize(self, strategy_context: Any):
        """Initialize technical indicators"""
        import backtrader as bt
        
        data_feed = strategy_context.datas[0]
        
        self.indicators = {
            'sma_short': bt.indicators.SimpleMovingAverage(data_feed, period=10),
            'sma_long': bt.indicators.SimpleMovingAverage(data_feed, period=30),
            'rsi': bt.indicators.RelativeStrengthIndex(data_feed, period=14),
            'macd': bt.indicators.MACD(data_feed),
            'bb': bt.indicators.BollingerBands(data_feed, period=20)
        }
    
    def generate_signal(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """Generate signal based on technical indicators"""
        try:
            signal_score = 0.5  # Start neutral
            confidence = 0.7
            
            # Moving average crossover
            if not np.isnan(self.indicators['sma_short'][0]) and not np.isnan(self.indicators['sma_long'][0]):
                if self.indicators['sma_short'][0] > self.indicators['sma_long'][0]:
                    signal_score += 0.2
                else:
                    signal_score -= 0.2
            
            # RSI signals
            if not np.isnan(self.indicators['rsi'][0]):
                rsi = self.indicators['rsi'][0]
                if rsi < 30:  # Oversold
                    signal_score += 0.2
                elif rsi > 70:  # Overbought
                    signal_score -= 0.2
            
            # MACD signals
            if (not np.isnan(self.indicators['macd'].macd[0]) and 
                not np.isnan(self.indicators['macd'].signal[0])):
                if self.indicators['macd'].macd[0] > self.indicators['macd'].signal[0]:
                    signal_score += 0.1
                else:
                    signal_score -= 0.1
            
            # Bollinger Bands
            close_price = data.get('close', 0)
            if (not np.isnan(self.indicators['bb'].lines.bot[0]) and 
                not np.isnan(self.indicators['bb'].lines.top[0])):
                if close_price < self.indicators['bb'].lines.bot[0]:
                    signal_score += 0.1  # Near lower band - potential buy
                elif close_price > self.indicators['bb'].lines.top[0]:
                    signal_score -= 0.1  # Near upper band - potential sell
            
            # Clamp signal to [0, 1] range
            signal_score = max(0, min(1, signal_score))
            
            return signal_score, confidence
            
        except Exception as e:
            print(f"Error in technical indicator signal generation: {e}")
            return 0.5, 0.5


class RandomSignalGenerator(SignalGenerator):
    """
    Random signal generator for testing purposes
    """
    
    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(seed)
    
    def initialize(self, strategy_context: Any):
        """No initialization needed for random signals"""
        pass
    
    def generate_signal(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """Generate random signals"""
        signal = np.random.uniform(0.2, 0.8)
        confidence = np.random.uniform(0.5, 0.9)
        return signal, confidence


class PrecomputedSignalGenerator(SignalGenerator):
    """
    Signal generator that uses pre-computed signals from data feed
    """
    
    def __init__(self, signal_col: str = 'signal', confidence_col: str = 'confidence', **kwargs):
        super().__init__(**kwargs)
        self.signal_col = signal_col
        self.confidence_col = confidence_col
        self.data_feed = None
    
    def initialize(self, strategy_context: Any):
        """Store reference to data feed"""
        self.data_feed = strategy_context.datas[0]
    
    def generate_signal(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """Get signal from pre-computed columns in data feed"""
        try:
            # Try to get from data dict first
            if self.signal_col in data and self.confidence_col in data:
                return data[self.signal_col], data[self.confidence_col]
            
            # Try to get from data feed
            if self.data_feed and hasattr(self.data_feed, self.signal_col):
                signal = getattr(self.data_feed, self.signal_col)[0]
                confidence = getattr(self.data_feed, self.confidence_col)[0] if hasattr(self.data_feed, self.confidence_col) else 0.7
                return signal, confidence
            
            return 0.5, 0.5
            
        except (IndexError, AttributeError, KeyError):
            return 0.5, 0.5


def create_signal_generator(generator_type: str, **kwargs) -> SignalGenerator:
    """
    Factory function to create signal generators
    
    Args:
        generator_type: Type of generator ('ml', 'technical', 'random', 'precomputed')
        **kwargs: Additional parameters for the generator
        
    Returns:
        SignalGenerator instance
    """
    generators = {
        'ml': MLClassificationSignalGenerator,
        'technical': TechnicalIndicatorSignalGenerator,
        'random': RandomSignalGenerator,
        'precomputed': PrecomputedSignalGenerator
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. Available: {list(generators.keys())}")
    
    return generators[generator_type](**kwargs)
