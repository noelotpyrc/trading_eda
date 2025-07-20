# Polymarket Fat Tail Prediction System

## 🎯 Objective
Build a systematic approach to identify and capitalize on mispriced low probability events on Polymarket by collecting historical data, monitoring current markets, and predicting fat tail outcomes.

## 📊 Data Collection Strategy

### Target Markets
- **Historical**: All resolved markets with volume > $10,000
- **Current**: Active markets with volume > $1,000
- **Focus**: Fat tail events (outcomes <10% or >90% probability)

### API Endpoints
**Base URL**: `https://gamma-api.polymarket.com`

1. **GET /events** - Event-level data with volume/liquidity filters
2. **GET /markets** - Market-level data with resolution status  
3. **GET /prices-history** - Historical price data for specific markets
4. **GET /trades** - Individual trade data via CLOB API

### Key API Parameters
```python
# Historical resolved markets
historical_params = {
    'closed': True,           # Resolved markets only
    'volume_min': 10000,      # Minimum $10K volume
    'limit': 1000,            # Batch size for pagination
    'order': 'volume',        # Sort by volume descending
    'ascending': False
}

# Active market monitoring  
live_params = {
    'active': True,           # Only active markets
    'volume_min': 1000,       # Lower threshold for monitoring
    'order': 'start_date',    # Recent markets first
    'ascending': False
}
```

## 🛠️ System Architecture

### 1. Historical Data Collector (`polymarket_historical.py`)
**Purpose**: Build training dataset from resolved markets

**Features**:
- Fetch all resolved markets above volume threshold
- Collect complete price history for each market
- Extract market metadata (category, timeframe, initial odds)
- Identify fat tail events (extreme outcomes)
- Store structured dataset for model training

**Data Schema**:
```python
{
    'market_id': str,
    'event_id': str, 
    'question': str,
    'category': str,
    'volume': float,
    'liquidity': float,
    'start_date': datetime,
    'end_date': datetime,
    'resolution_date': datetime,
    'initial_price': float,
    'final_outcome': float,  # 0 or 1
    'price_history': List[dict],
    'is_fat_tail': bool,     # <0.1 or >0.9 initial price
    'volume_pattern': dict,   # Daily volume progression
    'price_volatility': float
}
```

### 2. Live Market Monitor (`polymarket_live.py`)
**Purpose**: Real-time monitoring for trading opportunities

**Features**:
- Monitor active markets continuously
- Track price movements and volume changes
- Calculate edge based on historical model predictions
- Send alerts for high-value opportunities
- Log market data for model updates

**Alert Criteria**:
- Model predicts fat tail event with >60% confidence
- Current market price offers >20% edge
- Sufficient liquidity for meaningful position
- Time to resolution >24 hours (avoid last-minute volatility)

### 3. Fat Tail Predictor (`fat_tail_model.py`)
**Purpose**: ML model to predict extreme outcomes

**Model Architecture**:
- **Input Features**: Market characteristics, volume patterns, price dynamics
- **Target**: Binary classification (fat tail vs normal outcome)
- **Algorithm**: Gradient boosting (XGBoost/LightGBM) for feature importance
- **Validation**: Time-series split to prevent lookahead bias

**Feature Engineering**:
```python
# Market Characteristics
- category_encoded: str          # Politics, Sports, Crypto, etc.
- days_to_resolution: int        # Time horizon
- initial_price: float           # Starting probability
- question_complexity: int       # Word count, question type

# Volume Patterns  
- early_volume_ratio: float      # First 24h volume / total
- volume_acceleration: float     # Volume growth rate
- liquidity_depth: float         # Order book depth
- unique_traders: int            # Number of participants

# Price Dynamics
- price_volatility: float        # Standard deviation of prices
- momentum_1d: float             # 1-day price change
- momentum_7d: float             # 7-day price change
- mean_reversion: float          # Price vs moving average

# External Signals
- related_market_correlation: float  # Correlation with similar markets
- news_sentiment: float              # Sentiment analysis of related news
- social_mentions: int               # Social media activity
```

**Output**:
- `fat_tail_probability`: float (0-1)
- `confidence_score`: float (model uncertainty)
- `recommended_position`: float (Kelly Criterion sizing)

## 🧠 Fat Tail Model Strategy

### Definition of Fat Tail Events
- **Low Probability**: Initial market price <20% that resolved YES
- **High Probability**: Initial market price >80% that resolved NO
- **Focus**: Markets where consensus was significantly wrong

### Training Approach
1. **Feature Selection**: Use historical data to identify predictive signals
2. **Class Balancing**: Oversample fat tail events (rare class)
3. **Cross-Validation**: Time-series splits to prevent data leakage
4. **Hyperparameter Tuning**: Optimize for precision on fat tail class

### Risk Management
- **Position Sizing**: Kelly Criterion based on edge and confidence
- **Diversification**: Maximum 5% of bankroll per market
- **Stop Loss**: Exit if price moves against position by 50%
- **Time Decay**: Reduce position size as resolution approaches

## 📈 Backtesting Framework

### Historical Performance Metrics
- **Win Rate**: Percentage of profitable fat tail predictions
- **Expected Value**: Average return per prediction
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst losing streak
- **Kelly Bankroll Growth**: Compound annual growth rate

### Market Category Analysis
- **Politics**: Election outcomes, policy predictions
- **Sports**: Underdog victories, record-breaking events  
- **Crypto**: Price predictions, adoption milestones
- **Economics**: Recession timing, inflation targets

### Signal Effectiveness
- **Volume Patterns**: Early volume surge predictive power
- **Price Momentum**: Trend continuation vs reversal
- **Market Sentiment**: Contrarian vs momentum strategies
- **External Data**: News, social media, related markets

## 🚀 Implementation Timeline

### Phase 1: Data Collection (Week 1-2)
1. Set up API connections and authentication
2. Build historical data collector
3. Collect 2+ years of resolved market data
4. Clean and structure dataset

### Phase 2: Model Development (Week 3-4) 
1. Exploratory data analysis on historical data
2. Feature engineering and selection
3. Train initial fat tail prediction model
4. Validate on out-of-sample data

### Phase 3: Live Trading (Week 5-6)
1. Deploy live market monitor
2. Integrate model predictions with real-time data
3. Implement risk management and position sizing
4. Start paper trading to validate system

### Phase 4: Optimization (Ongoing)
1. Collect live trading results
2. Retrain model with new data
3. Refine feature engineering
4. Optimize risk management parameters

## 🔧 Technical Requirements

### Dependencies
```python
- requests          # API calls
- pandas           # Data manipulation  
- numpy            # Numerical computations
- scikit-learn     # ML models and preprocessing
- xgboost          # Gradient boosting
- matplotlib       # Plotting
- seaborn          # Statistical visualization
- jupyter          # Analysis notebooks
- python-dotenv    # Environment variables
- schedule         # Task scheduling
```

### Infrastructure
- **Database**: SQLite for local development, PostgreSQL for production
- **Scheduling**: Cron jobs for data collection, live monitoring
- **Alerting**: Email/Slack notifications for trading opportunities
- **Logging**: Comprehensive logging for debugging and analysis
- **Monitoring**: System health checks and performance metrics

## 📊 Expected Outcomes

### Success Metrics
- **Model Accuracy**: >65% precision on fat tail prediction
- **Trading Performance**: >15% annual return with <20% max drawdown
- **Market Coverage**: Monitor 100+ active markets simultaneously
- **Data Quality**: 95%+ successful API calls and data completeness

### Risk Factors
- **API Rate Limits**: Polymarket may restrict high-frequency requests
- **Market Manipulation**: Large players may move prices artificially
- **Model Overfitting**: Historical patterns may not persist
- **Regulatory Risk**: Prediction market regulations may change

### Mitigation Strategies
- **API Management**: Implement exponential backoff and request queuing
- **Outlier Detection**: Filter suspicious price movements
- **Regular Retraining**: Update model monthly with new data
- **Diversification**: Trade multiple market categories and timeframes

## 📝 Next Steps

1. **Create script scaffolds** for the three main components
2. **Set up development environment** with required dependencies
3. **Test API connectivity** and understand rate limits
4. **Start collecting historical data** to build initial dataset
5. **Begin exploratory analysis** to understand market patterns

This plan provides a systematic approach to identifying and profiting from mispriced fat tail events on Polymarket through data-driven analysis and machine learning.