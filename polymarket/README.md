# Polymarket Fat Tail Prediction System

A systematic approach to identify and capitalize on mispriced low probability events on Polymarket using machine learning and data-driven analysis.

## 🎯 Overview

This system consists of three main components:

1. **Historical Data Collector** (`polymarket_historical.py`) - Builds training dataset from resolved markets
2. **Live Market Monitor** (`polymarket_live.py`) - Real-time monitoring for trading opportunities  
3. **Fat Tail Predictor** (`fat_tail_model.py`) - ML model to predict extreme outcomes

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd polymarket/

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Historical Data

```bash
# Collect resolved markets with >$10K volume
python polymarket_historical.py \
    --volume_min 10000 \
    --output_dir ./data \
    --format all
```

### 3. Train Fat Tail Model

```bash
# Train models on historical data
python fat_tail_model.py \
    --data_path ./data/polymarket_historical.csv \
    --output_dir ./models \
    --train_split 0.8
```

### 4. Start Live Monitoring

```bash
# Monitor active markets for opportunities
python polymarket_live.py \
    --model_path ./models/fat_tail_model_xgboost.pkl \
    --volume_min 1000 \
    --check_interval 300
```

## 📊 System Architecture

### Data Collection Strategy
- **Historical**: Resolved markets with volume > $10K for training
- **Live**: Active markets with volume > $1K for monitoring
- **Focus**: Fat tail events (outcomes <10% or >90% initial probability)

### Feature Engineering
- **Market characteristics**: Volume, liquidity, duration, category
- **Price dynamics**: Volatility, momentum, mean reversion
- **Question analysis**: Complexity, keywords, structure
- **Volume patterns**: Early volume, acceleration, peak timing
- **Temporal features**: Month, day of week, time to resolution

### Model Architecture
- **Primary**: XGBoost classifier for fat tail prediction
- **Alternatives**: Random Forest, Logistic Regression
- **Validation**: Time-series cross-validation to prevent lookahead bias
- **Metrics**: ROC-AUC, precision/recall on fat tail class

### Risk Management
- **Position sizing**: Kelly Criterion based on edge and confidence
- **Diversification**: Maximum 5% of bankroll per market
- **Time limits**: Minimum 24 hours to resolution
- **Liquidity**: Minimum $1000 market liquidity

## 📁 File Structure

```
polymarket/
├── PLAN.md                      # Detailed system documentation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── polymarket_historical.py     # Historical data collector
├── polymarket_live.py          # Live market monitor
├── fat_tail_model.py           # ML model training/prediction
├── data/                       # Data storage
│   ├── polymarket_historical.csv
│   ├── polymarket_historical.json
│   └── polymarket_historical.parquet
├── models/                     # Trained models
│   ├── fat_tail_model_xgboost.pkl
│   └── training_results.json
└── logs/                       # Application logs
    ├── polymarket_historical.log
    ├── polymarket_live.log
    └── fat_tail_model.log
```

## 🔧 Configuration

### API Settings
The system uses Polymarket's Gamma API:
- **Base URL**: `https://gamma-api.polymarket.com`
- **Rate limiting**: Built-in exponential backoff
- **No authentication required** for public endpoints

### Model Parameters
Key parameters for fat tail detection:
- **Confidence threshold**: 60% minimum model confidence
- **Edge threshold**: 15% minimum price edge
- **Position sizing**: Kelly Criterion with 5% maximum
- **Time horizon**: 24+ hours to resolution

### Alert Criteria
Opportunities trigger alerts when:
- Model predicts fat tail with >60% confidence
- Current market price offers >20% edge
- Sufficient liquidity for meaningful position
- Time to resolution >24 hours

## 📈 Expected Performance

### Success Metrics
- **Model accuracy**: >65% precision on fat tail prediction
- **Trading performance**: >15% annual return with <20% max drawdown
- **Market coverage**: Monitor 100+ active markets simultaneously

### Risk Factors
- **API rate limits**: Polymarket may restrict high-frequency requests
- **Market manipulation**: Large players may move prices artificially
- **Model overfitting**: Historical patterns may not persist
- **Regulatory risk**: Prediction market regulations may change

## 🛠️ Development

### Running Tests
```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .
```

### Adding Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## 📊 Data Sources

### Polymarket API Endpoints
- **GET /events** - Event-level data with volume filters
- **GET /markets** - Market-level data with resolution status
- **GET /prices-history** - Historical price data
- **GET /trades** - Individual trade data

### Data Schema
Each market record contains:
- Market metadata (ID, question, category, dates)
- Volume and liquidity metrics
- Price history and volatility
- Resolution outcome
- Fat tail classification

## 🔍 Model Interpretation

### Feature Importance
Top predictive features typically include:
1. **Initial price extremity** - Distance from 50%
2. **Market category** - Politics vs Sports vs Crypto
3. **Time to resolution** - Longer markets more volatile
4. **Early volume patterns** - Volume concentration timing
5. **Question complexity** - Word count and structure

### SHAP Analysis
Use SHAP values to understand individual predictions:
```python
import shap

# Load trained model
model = FatTailModel.load_model('models/fat_tail_model_xgboost.pkl')

# Explain predictions
explainer = shap.Explainer(model.models['xgboost']['model'])
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])
```

## 📞 Support

### Logging
All components generate detailed logs:
- `polymarket_historical.log` - Data collection progress
- `polymarket_live.log` - Live monitoring alerts
- `fat_tail_model.log` - Model training metrics

### Database
Live monitor uses SQLite database (`polymarket_live.db`) for:
- Market snapshots over time
- Trading opportunity history
- Performance tracking

### Troubleshooting
Common issues and solutions:
1. **API rate limits** - Increase delays between requests
2. **Model loading errors** - Check file paths and dependencies
3. **Database locks** - Ensure single process access
4. **Memory issues** - Reduce batch sizes for large datasets

## 🚀 Future Enhancements

### Planned Features
- **Sentiment analysis** - News and social media signals
- **Multi-market correlation** - Related market signals
- **Options strategies** - Complex position structures
- **Automated trading** - Direct market execution
- **Portfolio optimization** - Multi-market position sizing

### Research Areas
- **Ensemble models** - Combine multiple prediction approaches
- **Deep learning** - LSTM/Transformer models for sequences
- **Reinforcement learning** - Dynamic strategy optimization
- **Alternative data** - Weather, economic indicators, etc.

## 📄 License

This project is for educational and research purposes. Please ensure compliance with Polymarket's terms of service and applicable regulations before any commercial use.

## 🤝 Contributing

Contributions welcome! Please read the development guidelines and submit pull requests for review.