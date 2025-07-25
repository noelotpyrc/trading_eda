#!/usr/bin/env python3
"""
Backtesting Package
Trading strategy backtesting framework for onchain and traditional assets

CURRENTLY ACTIVE MODULES:
========================
- solana_transaction_feed: Onchain transaction data feed and strategy
- onchain_broker: Custom broker for onchain transaction execution  
- base_strategy: Abstract base classes for strategy development

FUTURE MODULES (STUBBED):
========================
- signal_generators: Multi-signal ensemble framework (TODO)
- data_utils: OHLCV data loading and preprocessing (TODO)
- backtest_runner: Production backtesting system (TODO)
- ml_strategy: ML-based strategies for traditional assets (TODO)

USAGE:
======
For onchain transaction strategies:
    from backtesting.solana_transaction_feed import SolanaTransactionFeed, SolanaTransactionStrategy
    from backtesting.onchain_broker import setup_onchain_broker

For custom strategy development:
    from backtesting.base_strategy import BaseStrategy
"""

# Currently active exports (used by examples)
from .solana_transaction_feed import SolanaTransactionFeed
from .onchain_broker import setup_onchain_broker, OnchainBroker, OnchainCommissionInfo
from .base_strategy import BaseStrategy, SignalBasedStrategy
from .strategies import SolanaTransactionStrategy, DiagnosticSolanaStrategy

# Future exports (currently stubbed - will raise NotImplementedError)
# Uncomment when implementing:
# from .signal_generators import SignalGenerator
# from .data_utils import load_ohlcv_data, prepare_data_for_backtest  
# from .backtest_runner import BacktestRunner, quick_backtest
# from .ml_strategy import MLClassificationStrategy, MLDataFeed

__version__ = "0.1.0"

__all__ = [
    # Currently active
    "SolanaTransactionFeed",
    "SolanaTransactionStrategy",
    "DiagnosticSolanaStrategy", 
    "setup_onchain_broker",
    "OnchainBroker",
    "OnchainCommissionInfo",
    "BaseStrategy",
    "SignalBasedStrategy",
    
    # Future (stubbed)
    # "SignalGenerator",
    # "load_ohlcv_data", 
    # "prepare_data_for_backtest",
    # "BacktestRunner",
    # "quick_backtest",
    # "MLClassificationStrategy",
    # "MLDataFeed",
]