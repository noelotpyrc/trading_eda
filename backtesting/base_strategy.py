#!/usr/bin/env python3
"""
Base Strategy Classes for Backtrader
Provides abstract framework for different trading strategies
"""

import backtrader as bt
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any


class BaseStrategy(bt.Strategy, ABC):
    """
    Abstract base class for all trading strategies
    """
    
    params = (
        ('stop_loss', 0.05),          # Stop loss percentage
        ('take_profit', 0.15),        # Take profit percentage
        ('position_size', 0.95),      # Percentage of available cash to use
        ('lookback_window', 30),      # Days to lookback for calculations
        ('log_frequency', 10),        # Log every N bars
        ('verbose', True),            # Enable logging
    )
    
    def __init__(self):
        """Initialize base strategy components"""
        # Reference to data
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        
        # Track orders and positions
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        
        # Initialize strategy-specific components
        self._initialize_strategy()
        
        if self.params.verbose:
            self._log_initialization()
    
    @abstractmethod
    def _initialize_strategy(self):
        """Initialize strategy-specific components (must be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _get_signal(self) -> Tuple[float, float]:
        """Get trading signal and confidence (must be implemented by subclasses)
        
        Returns:
            Tuple of (signal_strength, confidence) where:
            - signal_strength: 0-1 value (0=strong sell, 0.5=neutral, 1=strong buy)
            - confidence: 0-1 value indicating prediction confidence
        """
        pass
    
    def _log_initialization(self):
        """Log strategy initialization"""
        print(f"Strategy {self.__class__.__name__} initialized with parameters:")
        for param_name in self.params._getkeys():
            value = getattr(self.params, param_name)
            if isinstance(value, float) and param_name in ['stop_loss', 'take_profit', 'position_size']:
                print(f"  {param_name}: {value * 100:.1f}%")
            else:
                print(f"  {param_name}: {value}")
    
    def notify_order(self, order):
        """Notification of order status"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
                if self.params.verbose:
                    print(f'BUY EXECUTED: Price: {order.executed.price:.2f}, '
                          f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:  # Sell
                if self.params.verbose:
                    print(f'SELL EXECUTED: Price: {order.executed.price:.2f}, '
                          f'Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Calculate P&L
                if self.buy_price:
                    pnl = (order.executed.price - self.buy_price) * order.executed.size
                    pnl_pct = (order.executed.price - self.buy_price) / self.buy_price * 100
                    if self.params.verbose:
                        print(f'P&L: {pnl:.2f} ({pnl_pct:.2f}%)')
                    
                    if pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                    
                    self.trade_count += 1
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.params.verbose:
                print(f'Order {order.status}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Notification of trade completion"""
        if not trade.isclosed:
            return
        
        if self.params.verbose:
            print(f'TRADE PROFIT: GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
    
    def next(self):
        """Main strategy logic - called for each bar"""
        current_date = self.datas[0].datetime.date(0)
        
        # Skip if insufficient data
        if len(self.datas[0]) < self.params.lookback_window:
            return
        
        # Check if we have a pending order
        if self.order:
            return
        
        # Get signal from subclass implementation
        signal_strength, confidence = self._get_signal()
        
        # Current position and portfolio state
        current_position = self.position.size
        current_cash = self.broker.get_cash()
        current_value = self.broker.get_value()
        
        # Log current state periodically
        if self.params.verbose and len(self.datas[0]) % self.params.log_frequency == 0:
            print(f'Date: {current_date}, Close: {self.dataclose[0]:.2f}, '
                  f'Signal: {signal_strength:.3f}, Confidence: {confidence:.3f}, '
                  f'Position: {current_position}, Value: {current_value:.2f}')
        
        # Execute trading logic
        if not current_position:  # Not in position
            if self._should_buy(signal_strength, confidence):
                size = self._calculate_position_size()
                if size > 0:
                    self.order = self.buy(size=size)
                    if self.params.verbose:
                        print(f'BUY ORDER: {size} shares at {self.dataclose[0]:.2f}, '
                              f'Signal: {signal_strength:.3f}, Confidence: {confidence:.3f}')
        
        else:  # In position
            sell_reason = self._should_sell(signal_strength, confidence)
            if sell_reason:
                self.order = self.sell()
                if self.params.verbose:
                    print(f'SELL ORDER: {current_position} shares at {self.dataclose[0]:.2f}, '
                          f'Reason: {sell_reason}')
    
    def _should_buy(self, signal_strength: float, confidence: float) -> bool:
        """Determine if we should buy (can be overridden by subclasses)"""
        return signal_strength > 0.6 and confidence > 0.7
    
    def _should_sell(self, signal_strength: float, confidence: float) -> Optional[str]:
        """Determine if we should sell and return reason (can be overridden by subclasses)"""
        if self.buy_price is None:
            return None
        
        current_pnl_pct = (self.dataclose[0] - self.buy_price) / self.buy_price
        
        # Stop loss
        if current_pnl_pct <= -self.params.stop_loss:
            return f"Stop Loss ({current_pnl_pct*100:.1f}%)"
        
        # Take profit
        if current_pnl_pct >= self.params.take_profit:
            return f"Take Profit ({current_pnl_pct*100:.1f}%)"
        
        # Signal-based sell (low signal with high confidence)
        if signal_strength < 0.4 and confidence > 0.7:
            return f"Signal Sell ({signal_strength:.3f})"
        
        return None
    
    def _calculate_position_size(self) -> int:
        """Calculate position size based on available cash"""
        available_cash = self.broker.get_cash() * self.params.position_size
        return int(available_cash / self.dataclose[0])
    
    def stop(self):
        """Called when strategy ends"""
        if not self.params.verbose:
            return
        
        final_value = self.broker.get_value()
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        
        print(f'\n=== STRATEGY COMPLETED ===')
        print(f'Strategy: {self.__class__.__name__}')
        print(f'Final Portfolio Value: {final_value:.2f}')
        print(f'Total Trades: {self.trade_count}')
        print(f'Wins: {self.win_count}, Losses: {self.loss_count}')
        print(f'Win Rate: {win_rate:.1f}%')


class SignalBasedStrategy(BaseStrategy):
    """
    Strategy that uses pre-computed signals from data feed
    """
    
    params = (
        ('signal_threshold', 0.6),     # Threshold for buy signals
        ('confidence_threshold', 0.7), # Minimum confidence required
    )
    
    def _initialize_strategy(self):
        """Initialize signal-based strategy"""
        # Check for signal columns in data feed
        self.signal_col = getattr(self.datas[0], 'signal', None)
        self.confidence_col = getattr(self.datas[0], 'confidence', None)
        
        if self.signal_col is None or self.confidence_col is None:
            print("⚠️  Warning: Signal or confidence columns not found in data feed")
    
    def _get_signal(self) -> Tuple[float, float]:
        """Get signal from pre-computed columns"""
        try:
            signal = self.signal_col[0] if self.signal_col is not None else 0.5
            confidence = self.confidence_col[0] if self.confidence_col is not None else 0.5
            return signal, confidence
        except (IndexError, AttributeError):
            return 0.5, 0.5
    
    def _should_buy(self, signal_strength: float, confidence: float) -> bool:
        """Buy when signal exceeds threshold with sufficient confidence"""
        return (signal_strength >= self.params.signal_threshold and 
                confidence >= self.params.confidence_threshold)
