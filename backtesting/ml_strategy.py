#!/usr/bin/env python3
"""
ML Classification Strategy Implementation
Specific strategy implementation using machine learning signals
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from .base_strategy import BaseStrategy
from .signal_generators import create_signal_generator, SignalGenerator


class MLClassificationStrategy(BaseStrategy):
    """
    Trading strategy based on machine learning classification signals
    """
    
    params = (
        ('ml_threshold', 0.6),        # Probability threshold for buy signals
        ('min_confidence', 0.7),      # Minimum confidence for trading
        ('signal_generator_type', 'ml'),  # Type of signal generator to use
        ('model_path', None),         # Path to specific ML model (optional)
    )
    
    def _initialize_strategy(self):
        """Initialize ML-specific strategy components"""
        # Create signal generator
        generator_config = {
            'model_path': self.params.model_path
        }
        
        self.signal_generator = create_signal_generator(
            self.params.signal_generator_type, 
            **generator_config
        )
        
        # Initialize the generator with strategy context
        self.signal_generator.initialize(self)
        
        # Additional technical indicators for feature calculation
        self.sma_short = bt.indicators.SimpleMovingAverage(self.datas[0], period=10)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.datas[0], period=30)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.datas[0], period=14)
        self.macd = bt.indicators.MACD(self.datas[0])
    
    def _get_signal(self) -> Tuple[float, float]:
        """Get ML prediction for current market state"""
        # Prepare current market data
        market_data = {
            'close': self.dataclose[0],
            'open': self.dataopen[0],
            'high': self.datahigh[0],
            'low': self.datalow[0],
            'volume': self.datavolume[0],
        }
        
        # Add previous close if available
        if len(self.datas[0]) > 1:
            market_data['prev_close'] = self.dataclose[-1]
        
        # Check for pre-computed signals in data feed
        if hasattr(self.datas[0], 'ml_signal'):
            try:
                market_data['ml_signal'] = self.datas[0].ml_signal[0]
                market_data['ml_confidence'] = self.datas[0].ml_confidence[0]
            except (IndexError, AttributeError):
                pass
        
        # Generate signal using the signal generator
        return self.signal_generator.generate_signal(market_data)
    
    def _should_buy(self, signal_strength: float, confidence: float) -> bool:
        """ML-specific buy conditions"""
        return (signal_strength >= self.params.ml_threshold and 
                confidence >= self.params.min_confidence)
    
    def _should_sell(self, signal_strength: float, confidence: float) -> str:
        """ML-specific sell conditions with base risk management"""
        # First check base risk management rules
        base_sell_reason = super()._should_sell(signal_strength, confidence)
        if base_sell_reason:
            return base_sell_reason
        
        # ML-specific sell signal (low probability with high confidence)
        if (signal_strength <= (1 - self.params.ml_threshold) and 
            confidence >= self.params.min_confidence):
            return f"ML Sell Signal ({signal_strength:.3f})"
        
        return None


class MLDataFeed(bt.feeds.PandasData):
    """
    Custom data feed that includes ML predictions
    """
    lines = ('ml_signal', 'ml_confidence',)
    
    params = (
        ('ml_signal', -1),     # Column index for ML signal
        ('ml_confidence', -1), # Column index for ML confidence
    )


class MultiSignalStrategy(BaseStrategy):
    """
    Strategy that combines multiple signal sources
    """
    
    params = (
        ('signal_weights', None),     # Dictionary of signal weights
        ('ensemble_threshold', 0.6),  # Threshold for ensemble signal
        ('min_confidence', 0.7),      # Minimum confidence required
    )
    
    def _initialize_strategy(self):
        """Initialize multiple signal generators"""
        # Default signal weights if not provided
        if self.params.signal_weights is None:
            self.signal_weights = {
                'ml': 0.6,
                'technical': 0.4
            }
        else:
            self.signal_weights = self.params.signal_weights
        
        # Create signal generators
        self.signal_generators = {}
        for signal_type, weight in self.signal_weights.items():
            if weight > 0:  # Only create generators with positive weights
                self.signal_generators[signal_type] = create_signal_generator(signal_type)
                self.signal_generators[signal_type].initialize(self)
    
    def _get_signal(self) -> Tuple[float, float]:
        """Combine signals from multiple generators"""
        # Prepare market data
        market_data = {
            'close': self.dataclose[0],
            'open': self.dataopen[0],
            'high': self.datahigh[0],
            'low': self.datalow[0],
            'volume': self.datavolume[0],
        }
        
        if len(self.datas[0]) > 1:
            market_data['prev_close'] = self.dataclose[-1]
        
        # Get signals from all generators
        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for signal_type, generator in self.signal_generators.items():
            weight = self.signal_weights[signal_type]
            signal, confidence = generator.generate_signal(market_data)
            
            weighted_signal += signal * weight
            weighted_confidence += confidence * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_signal = weighted_signal / total_weight
            ensemble_confidence = weighted_confidence / total_weight
        else:
            ensemble_signal = 0.5
            ensemble_confidence = 0.5
        
        return ensemble_signal, ensemble_confidence
    
    def _should_buy(self, signal_strength: float, confidence: float) -> bool:
        """Buy based on ensemble signal"""
        return (signal_strength >= self.params.ensemble_threshold and 
                confidence >= self.params.min_confidence)
