# Solana Trading Data - Comprehensive Data Dictionary

## Overview
This document provides a complete data dictionary for all tables, features, and data structures in the Solana trading analysis project. It consolidates schema information, feature definitions, and usage guidance across the entire data pipeline.

**Data Coverage**: 325M+ trades, 10M+ traders, 5.9K tokens  
**Time Period**: March 2021 - June 2025  
**Database**: DuckDB at `/Volumes/Extreme SSD/DuckDB/solana.duckdb`

---

## 📊 Core Database Tables

### **1. first_day_trades** - Primary Trading Data
**Records**: 325,171,663  
**Purpose**: Raw transaction data for first-day trading of new token launches  
**Date Range**: 2021-03-03 to 2025-06-15

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `mint` | VARCHAR | Token contract address (target asset) | `"ABC123...DEF789"` |
| `block_timestamp` | TIMESTAMP WITH TIME ZONE | Transaction timestamp | `2024-12-02 13:38:19-05:00` |
| `succeeded` | BOOLEAN | Transaction success status | `TRUE` |
| `swapper` | VARCHAR | Trader wallet address | `"trader123...xyz789"` |
| `swap_from_amount` | DOUBLE | Amount of source token | `100.5` |
| `swap_from_mint` | VARCHAR | Source token address | `"So111...112"` (SOL) |
| `swap_to_amount` | DOUBLE | Amount of destination token | `1000000.0` |
| `swap_to_mint` | VARCHAR | Destination token address | `"ABC123...DEF789"` |
| `__row_index` | BIGINT | Unique row identifier | `12345678` |

**Data Quality Notes**:
- Only successful trades included (`succeeded = TRUE`)
- SOL mint: `"So11111111111111111111111111111111111111112"`
- Supports 3 trade types: SOL→Token, Token→SOL, Token→Token

### **2. coin_first_two_hours** - Early Trading Window
**Records**: 133,394,160  
**Purpose**: First 2 hours of trading data for new coins (subset of first_day_trades)  
**Schema**: Identical to `first_day_trades`

**Usage**: Focused analysis on initial launch trading patterns

### **3. trader_features** - Behavioral Profiles
**Records**: 10,060,972  
**Purpose**: Aggregated trader-level behavioral and performance features

#### Volume & Scale Features (11 features)
| Feature | Type | Description | Calculation | Range |
|---------|------|-------------|-------------|-------|
| `swapper` | VARCHAR | Trader wallet address (PK) | - | - |
| `total_trades_count` | BIGINT | Total number of trades | COUNT(*) | 1 to 50,000+ |
| `total_sol_spent` | DOUBLE | Total SOL used for buying | SUM(swap_from_amount WHERE from_mint=SOL) | 0.001 to 100,000+ |
| `total_sol_received` | DOUBLE | Total SOL received from selling | SUM(swap_to_amount WHERE to_mint=SOL) | 0 to 100,000+ |
| `avg_sol_trade_size` | DOUBLE | Average SOL per trade | AVG(sol_amount) | 0.001 to 10,000+ |
| `median_sol_trade_size` | DOUBLE | Median SOL per trade | PERCENTILE_CONT(0.5) | 0.001 to 1,000+ |
| `max_single_sol_trade` | DOUBLE | Largest single trade | MAX(sol_amount) | 0.001 to 100,000+ |
| `min_sol_trade_size` | DOUBLE | Smallest trade size | MIN(sol_amount WHERE > 0) | 0.000001 to 100+ |
| `sol_trade_size_std_dev` | DOUBLE | Trade size volatility | STDDEV(sol_amount) | 0 to 10,000+ |
| `trade_size_coefficient_variation` | DOUBLE | Normalized volatility | std_dev / avg_trade_size | 0 to 50+ |
| `net_sol_pnl` | DOUBLE | Overall profit/loss | total_received - total_spent | -50,000 to +50,000 |

**Key Insights**:
- 49% of traders spend <1 SOL (retail segment)
- 0.3% spend >1K SOL (whale segment)
- High CV indicates inconsistent position sizing

#### Diversification & Specialization Features (3 features)
| Feature | Type | Description | Calculation | Range |
|---------|------|-------------|-------------|-------|
| `unique_coins_traded` | DOUBLE | Number of different tokens | COUNT(DISTINCT mint) | 1 to 500+ |
| `avg_trades_per_coin` | FLOAT | Trade concentration | total_trades / unique_coins | 1 to 1,000+ |
| `trade_concentration_ratio` | FLOAT | Focus on single coin | max_trades_single_coin / total_trades | 0.01 to 1.0 |

**Key Insights**:
- 56.7% trade only 1 coin (specialists)
- 0.2% trade >100 coins (diversifiers)

#### Timing & Behavioral Features (6 features)
| Feature | Type | Description | Calculation | Range |
|---------|------|-------------|-------------|-------|
| `trading_span_days` | DOUBLE | Days between first/last trade | (max_timestamp - min_timestamp) / 86400 | 0 to 1,500+ |
| `trades_per_day` | DOUBLE | Daily trading frequency | total_trades / trading_span_days | 0.001 to 10,000+ |
| `avg_hours_between_trades` | DOUBLE | Trade timing consistency | time_span / (trade_count - 1) / 3600 | 0.001 to 8,760+ |
| `active_hours` | BIGINT | Unique hours with trades | COUNT(DISTINCT hour_trunc) | 1 to 8,760+ |
| `active_days` | BIGINT | Unique days with trades | COUNT(DISTINCT day_trunc) | 1 to 1,500+ |
| `trades_per_active_hour` | FLOAT | Intensity when active | total_trades / active_hours | 1 to 1,000+ |

**Key Insights**:
- 7.5% trade <6min intervals (bot-like behavior)
- 28.9% have >24h between trades (casual traders)

#### Bot Detection Features (1 feature)
| Feature | Type | Description | Calculation | Range |
|---------|------|-------------|-------------|-------|
| `round_number_preference` | FLOAT | Preference for round amounts | % trades using round SOL amounts | 0.0 to 1.0 |

**Detection Logic**: Uses hardcoded list of round amounts (0.01, 0.1, 1, 5, 10, etc.)

#### Non-SOL Trade Analysis (9 features)
| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `sol_to_token_trades` | BIGINT | Buy trades count | 0 to 50,000+ |
| `token_to_sol_trades` | BIGINT | Sell trades count | 0 to 50,000+ |
| `token_to_token_trades` | BIGINT | Direct token swaps | 0 to 1,000+ |
| `unique_from_tokens_non_sol` | BIGINT | Source token diversity | 0 to 500+ |
| `unique_to_tokens_non_sol` | BIGINT | Target token diversity | 0 to 500+ |
| `sol_to_token_percentage` | FLOAT | Buy trade ratio | 0.0 to 1.0 |
| `token_to_sol_percentage` | FLOAT | Sell trade ratio | 0.0 to 1.0 |
| `token_to_token_percentage` | FLOAT | Arbitrage ratio | 0.0 to 1.0 |
| `buy_sell_ratio` | FLOAT | Buy/sell balance | 0.001 to 1,000+ |

**Key Insights**:
- 52.7% SOL→Token, 46.1% Token→SOL, 1.2% Token→Token
- 6.3% engage in token arbitrage

### **4. trader_coin_performance** - Position-Level Performance
**Records**: 43,441,089  
**Purpose**: Individual trader performance on each token

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `swapper` | VARCHAR | Trader wallet address | - |
| `mint` | VARCHAR | Token contract address | - |
| `sol_spent_on_coin` | DOUBLE | SOL invested in token | 0 to 100,000+ |
| `sol_received_from_coin` | DOUBLE | SOL returned from token | 0 to 100,000+ |
| `buy_trades` | BIGINT | Number of buy trades | 0 to 10,000+ |
| `sell_trades` | BIGINT | Number of sell trades | 0 to 10,000+ |
| `total_coin_trades` | BIGINT | Total trades on token | 1 to 20,000+ |
| `net_sol_pnl_per_coin` | DOUBLE | Profit/loss per token | -50,000 to +50,000 |
| `roi_on_coin` | DOUBLE | Return on investment % | -1.0 to 100+ |
| `hours_active_on_coin` | DOUBLE | Time spent trading token | 0.001 to 8,760+ |

**Performance Statistics**:
- 31.5% profitable positions, 64.2% losses
- Mean profit: +4.88 SOL, Mean loss: -2.60 SOL
- 27.9% positive ROI (excluding zero-spend positions)

### **5. trader_coin_first_two_hours_features** - Early Trading Features
**Records**: 17,585,219  
**Purpose**: Behavioral features during first 2 hours of coin trading

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `trader_id` | VARCHAR | Trader wallet address | - |
| `coin_id` | VARCHAR | Token contract address | - |
| `trade_count` | BIGINT | Number of trades in 2h | 1 to 1,000+ |
| `trades_per_hour` | DOUBLE | Trading frequency | 0.5 to 500+ |
| `time_span_minutes` | DOUBLE | Trading duration | 0.1 to 120 |
| `total_volume_traded` | DOUBLE | Total SOL volume | 0.001 to 10,000+ |
| `avg_trade_size` | DOUBLE | Average trade size | 0.001 to 10,000+ |
| `trade_size_cv` | DOUBLE | Trade size consistency | 0 to 10+ |
| `largest_trade_size` | DOUBLE | Maximum single trade | 0.001 to 10,000+ |
| `volume_concentration` | DOUBLE | Volume distribution | 0 to 10+ |
| `unique_trading_pairs` | BIGINT | Different token pairs | 1 to 50+ |
| `sol_involvement_ratio` | FLOAT | SOL-related trade ratio | 0.0 to 1.0 |
| `buy_ratio` | FLOAT | Buy vs sell preference | 0.0 to 1.0 |
| `round_number_ratio` | FLOAT | Round number usage | 0.0 to 1.0 |
| `trade_size_diversity` | FLOAT | Size variation | 0.0 to 1.0 |
| `avg_seconds_between_trades` | DOUBLE | Trading pace | 1 to 7,200 |
| `first_trade_minutes` | DOUBLE | Entry timing | 0 to 120 |

---

## 🔧 Feature Engineering Outputs

### **Classification Forward Features** - Time Series ML Features
**Generated by**: `batch_feature_extraction.sql`  
**Sampling**: 60-second intervals  
**Windows**: 30s, 60s, 120s lookback periods  
**Total Features**: 63 columns per sample

#### Core Feature Categories (20 features × 3 windows = 60 features)

**Volume Flow Features** (6 per window):
- `total_volume_{window}s`: Total SOL volume
- `buy_volume_{window}s`: Buying pressure
- `sell_volume_{window}s`: Selling pressure  
- `buy_ratio_{window}s`: Buy dominance (0-1)
- `volume_imbalance_{window}s`: Net flow direction (-1 to +1)
- `avg_txn_size_{window}s`: Average transaction size

**Transaction Flow Features** (5 per window):
- `total_txns_{window}s`: Transaction count
- `buy_txns_{window}s`: Buy transaction count
- `sell_txns_{window}s`: Sell transaction count
- `txn_buy_ratio_{window}s`: Transaction flow bias (0-1)
- `txn_flow_imbalance_{window}s`: Net transaction flow (-1 to +1)

**Trader Behavior Features** (4 per window):
- `unique_traders_{window}s`: Market participation breadth
- `unique_buyers_{window}s`: Buyer diversity
- `unique_sellers_{window}s`: Seller diversity
- `trader_buy_ratio_{window}s`: Trader sentiment bias (0-1)

**Position Size Distribution** (4 per window):
- `small_txn_ratio_{window}s`: Retail participation (0-1 SOL)
- `medium_txn_ratio_{window}s`: Small investor activity (1-10 SOL)
- `big_txn_ratio_{window}s`: Serious trader involvement (10-100 SOL)
- `whale_txn_ratio_{window}s`: Institutional activity (100+ SOL)

**Market Microstructure** (4 per window):
- `volume_per_trader_{window}s`: Average position size
- `volume_concentration_{window}s`: Volume inequality measure
- `volume_mean_{window}s`: Central tendency
- `volume_std_{window}s`: Volatility measure

#### Target Variables (3 features)
- `forward_buy_volume_300s`: Future buying pressure
- `forward_sell_volume_300s`: Future selling pressure
- `is_profitable_300s`: Binary profitability label (1=profitable, 0=not)

### **OHLC Financial Data** - Candlestick Format
**Generated by**: `aggregate_to_ohlc.sql`  
**Records**: 67,294,746  
**Aggregation**: By coin and block timestamp  
**Coverage**: 5,822 unique tokens

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `time` | VARCHAR | Human readable timestamp | `"20241202 13:38:19"` |
| `timestamp` | BIGINT | Unix timestamp (milliseconds) | 1618069845000+ |
| `open` | DOUBLE | First price in period | 0 to 9.90e+08 |
| `high` | DOUBLE | Highest price in period | 0 to 9.90e+08 |
| `low` | DOUBLE | Lowest price in period | 0 to 9.90e+08 |
| `close` | DOUBLE | Last price in period | 0 to 9.90e+08 |
| `vwap` | DOUBLE | Volume weighted average price | 0 to 9.90e+08 |
| `sol_volume` | DOUBLE | Total SOL traded | 0 to 46,719 |
| `token_volume` | DOUBLE | Total tokens traded | 0 to 132M+ |
| `transactions` | BIGINT | Number of trades aggregated | 1 to 454 |
| `mint` | VARCHAR | Token contract address | - |
| `block_timestamp_iso` | VARCHAR | ISO timestamp | `"2024-12-02 13:38:19-05:00"` |
| `otc` | VARCHAR | OTC flag (empty for compatibility) | `""` |
| `unique_prices` | BIGINT | Distinct prices in period | 1 to 50+ |
| `price_stddev` | DOUBLE | Price standard deviation | 0 to 1e+06 |

**Data Quality**:
- ✅ All OHLC relationships valid (high ≥ low, etc.)
- ⚠️ 68,399 zero-volume records (0.1% - rounding effects)
- 📊 Total volume: 721M SOL, 321M transactions

---

## 🔍 Data Quality Indicators

### **Completeness**
- **trader_features**: 100% complete for active traders
- **trader_coin_performance**: 9.4% missing ROI (zero-spend positions)
- **Classification features**: Varies by coin activity level

### **Data Ranges & Distributions**

**Volume Features** (Highly Skewed):
- Log-normal distribution
- Recommend log transformation
- Outliers: Whale traders (>1K SOL)

**Ratio Features** (Well-Bounded):
- Range: 0.0 to 1.0
- Generally well-distributed
- Safe for most ML algorithms

**Count Features** (Poisson-like):
- Heavy right tail
- Consider square root transformation
- Zero-inflation common

### **Known Issues**
1. **Price Extremes**: Some tokens show unrealistic prices (0 to 990M SOL)
2. **Zero Volumes**: Micro-transactions rounded to zero
3. **Missing Data**: Some early timestamp records incomplete
4. **Whale Skewness**: Volume distributions heavily skewed by large traders

---

## 🚀 Usage Guidelines

### **For Analysis**
1. **Start with**: `first_day_trades` for raw exploration
2. **Use**: `trader_features` for behavioral analysis
3. **Apply**: Classification features for ML modeling
4. **Leverage**: OHLC data for technical analysis

### **For Machine Learning**
```python
# Feature preprocessing example
features = ['total_volume_60s', 'buy_ratio_30s', 'whale_txn_ratio_120s']
X = np.log1p(data[volume_features])  # Log transform volumes
X = StandardScaler().fit_transform(X)  # Scale features
y = data['is_profitable_300s']  # Binary target
```

### **For Trading Strategy**
```sql
-- Whale activity detection
SELECT * FROM classification_features 
WHERE volume_per_trader_60s > 50 
  AND whale_txn_ratio_30s > 0.002
  AND avg_txn_size_60s > 4.0
-- Expected: 15.3% performance boost during these periods
```

### **Performance Optimization**
```sql
-- Recommended indexes
CREATE INDEX idx_trades_mint_timestamp ON first_day_trades(mint, block_timestamp);
CREATE INDEX idx_trades_swapper ON first_day_trades(swapper);
CREATE INDEX idx_trades_success ON first_day_trades(succeeded);
```

---

## 📋 Data Lineage

```
Raw Blockchain Data
    ↓
first_day_trades (325M records)
    ↓
├── trader_features (10M traders)
├── trader_coin_performance (43M positions)  
├── coin_first_two_hours (133M trades)
├── trader_coin_first_two_hours_features (17M features)
├── Classification Forward Features (Time series)
└── OHLC Financial Data (67M candles)
```

---

**Note**: This data dictionary serves as the definitive reference for all Solana trading data structures, feature definitions, and usage patterns across the analysis pipeline. For implementation details, refer to the specific script documentation in each module.