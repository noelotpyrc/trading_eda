# Solana Trader Profiling Features Documentation

## Overview
This feature engineering pipeline transforms raw Solana trading data into comprehensive trader-level behavioral and performance features for machine learning applications. The features are designed to capture distinct trading patterns, behaviors, and outcomes across multiple dimensions.

## Data Source
- **Base table**: `first_day_trades` (325M+ trades)
- **Scope**: 10.1M unique traders across 5.9K tokens
- **Time period**: First-day trading data for new coin launches

## Feature Categories

### 1. Volume & Scale Features (11 features)
**Purpose**: Capture trading volume patterns and position sizing behavior

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `total_trades_count` | Total number of trades | COUNT(*) |
| `total_sol_spent` | Total SOL used for buying | SUM(sol_from_amount) where from_mint = SOL |
| `total_sol_received` | Total SOL received from selling | SUM(sol_to_amount) where to_mint = SOL |
| `avg_sol_trade_size` | Average SOL per trade | AVG(sol_from_amount) for SOL trades |
| `median_sol_trade_size` | Median SOL per trade | PERCENTILE_CONT(0.5) for SOL trades |
| `max_single_sol_trade` | Largest single trade | MAX(sol_from_amount) |
| `min_sol_trade_size` | Smallest trade | MIN(sol_from_amount) where > 0 |
| `sol_trade_size_std_dev` | Trade size volatility | STDDEV(sol_from_amount) |
| `trade_size_coefficient_variation` | Normalized volatility | std_dev / avg_trade_size |
| `net_sol_pnl` | Overall profit/loss | total_sol_received - total_sol_spent |

**Key Insights**:
- 49% of traders spend <1 SOL (retail)
- 0.3% spend >1K SOL (whales)
- High CV indicates inconsistent position sizing

### 2. Diversification & Specialization Features (3 features)
**Purpose**: Measure multi-coin trading patterns and focus strategies

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `unique_coins_traded` | Number of different tokens | COUNT(DISTINCT mint) |
| `avg_trades_per_coin` | Trade concentration | total_trades / unique_coins |
| `trade_concentration_ratio` | Focus on single coin | max_trades_single_coin / total_trades |

**Key Insights**:
- 56.7% trade only 1 coin (specialists)
- 0.2% trade >100 coins (diversifiers)
- High concentration ratio = focused strategy

### 3. Timing & Behavioral Features (6 features)
**Purpose**: Analyze trading frequency, activity patterns, and temporal behavior

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `trading_span_days` | Days between first/last trade | (max_timestamp - min_timestamp) / 86400 |
| `trades_per_day` | Daily trading frequency | total_trades / trading_span_days |
| `avg_hours_between_trades` | Trade timing consistency | time_span / (trade_count - 1) / 3600 |
| `active_hours` | Unique hours with trades | COUNT(DISTINCT hour_trunc) |
| `active_days` | Unique days with trades | COUNT(DISTINCT day_trunc) |
| `trades_per_active_hour` | Intensity when active | total_trades / active_hours |

**Key Insights**:
- 7.5% trade <6min intervals (bot-like)
- 28.9% have >24h between trades (casual)
- High trades_per_hour suggests automation

### 4. Bot-like Behavior Features (1 feature)
**Purpose**: Detect automated trading patterns and sophistication

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `round_number_preference` | Preference for round amounts | % trades using round SOL amounts (0.01, 0.1, 1, 5, 10, etc.) |

**Detection Logic**:
- Hardcoded list of common round amounts
- Higher ratios suggest human trading
- Near-zero ratios suggest algorithmic precision

### 5. Non-SOL Trade Analysis (10 features)
**Purpose**: Capture token-to-token trading and arbitrage behavior

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `sol_to_token_trades` | Buy trades count | COUNT where from_mint = SOL |
| `token_to_sol_trades` | Sell trades count | COUNT where to_mint = SOL |
| `token_to_token_trades` | Direct token swaps | COUNT where neither mint = SOL |
| `unique_from_tokens_non_sol` | Source token diversity | COUNT(DISTINCT from_mint) excluding SOL |
| `unique_to_tokens_non_sol` | Target token diversity | COUNT(DISTINCT to_mint) excluding SOL |
| `sol_to_token_percentage` | Buy trade ratio | sol_to_token / total_trades |
| `token_to_sol_percentage` | Sell trade ratio | token_to_sol / total_trades |
| `token_to_token_percentage` | Arbitrage ratio | token_to_token / total_trades |
| `buy_sell_ratio` | Buy/sell balance | sol_to_token / token_to_sol |

**Key Insights**:
- 52.7% SOL→Token, 46.1% Token→SOL, 1.2% Token→Token
- 6.3% engage in token arbitrage
- High token diversity suggests sophisticated trading

### 6. Performance Features (Per-Coin Analysis)
**Purpose**: Track profit/loss performance at trader-coin level (43.4M records)

| Feature | Description | 
|---------|-------------|
| `sol_spent_on_coin` | SOL invested in specific token |
| `sol_received_from_coin` | SOL returned from token |
| `net_sol_pnl_per_coin` | Profit/loss per token |
| `roi_on_coin` | Return on investment % |
| `buy_trades` / `sell_trades` | Trade counts by direction |
| `hours_active_on_coin` | Time spent trading token |

**Key Performance Stats**:
- 31.5% profitable positions, 64.2% losses
- Mean profit: +4.88 SOL, Mean loss: -2.60 SOL
- 27.9% positive ROI (excluding zero-spend positions)

### 7. Aggregated Performance Features (4 features) from section 6
**Purpose**: Trader-level performance metrics aggregated from all positions

| Feature | Description | Calculation |
|---------|-------------|-------------|
| `total_positions` | Total number of coin positions | COUNT(*) from trader_coin_performance |
| `win_rate` | Percentage of profitable positions | SUM(net_pnl > 0) / COUNT(*) |
| `avg_pnl_per_position` | Average profit/loss per position | AVG(net_sol_pnl_per_coin) |
| `avg_roi` | Average return on investment | AVG(roi_on_coin) where roi_on_coin IS NOT NULL |

**Key Performance Stats**:
- Win rate ranges from 0% to 100%, with most traders having mixed results
- Average PnL per position varies widely, indicating skill differences
- 9.4% missing ROI data (positions with zero SOL spent)

## Data Quality & Completeness

### Missing Data Handling
- **ROI data**: 9.4% missing (zero SOL spent positions)
- **Round number preference**: Calculated only for SOL trades
- **Timing features**: Single-trade accounts have NULL intervals

### Feature Scaling Considerations
- **High skew**: Volume features need log transformation
- **Different scales**: SOL amounts vs. percentages vs. counts
- **Outliers**: Whale traders can skew distributions

## Usage Recommendations

### For Clustering
1. **Feature selection**: Focus on behavioral ratios rather than absolute volumes
2. **Preprocessing**: Log transform volume features, handle missing values strategically
3. **Key discriminators**: `trade_concentration_ratio`, `avg_hours_between_trades`, `token_to_token_percentage`

### For Classification
1. **Target definition**: Use performance features to create success labels
2. **Temporal features**: Critical for predicting bot vs. human behavior
3. **Interaction features**: Consider volume × timing combinations

### Feature Engineering Extensions
1. **Composite features**: Position velocity, diversification efficiency
2. **Temporal patterns**: Hour-of-day, day-of-week preferences
3. **Network features**: Co-trading patterns, following behavior

## Database Schema

### trader_features Table
```sql
CREATE TABLE trader_features (
    swapper VARCHAR,                    -- Wallet address
    total_trades_count BIGINT,          -- Volume features
    total_sol_spent DOUBLE,
    total_sol_received DOUBLE,
    avg_sol_trade_size DOUBLE,
    median_sol_trade_size DOUBLE,
    max_single_sol_trade DOUBLE,
    min_sol_trade_size DOUBLE,
    sol_trade_size_std_dev DOUBLE,
    trade_size_coefficient_variation DOUBLE,
    net_sol_pnl DOUBLE,
    unique_coins_traded DOUBLE,         -- Diversification features
    avg_trades_per_coin FLOAT,
    trade_concentration_ratio FLOAT,
    trading_span_days DOUBLE,           -- Timing features
    trades_per_day DOUBLE,
    avg_hours_between_trades DOUBLE,
    active_hours BIGINT,
    active_days BIGINT,
    trades_per_active_hour FLOAT,
    round_number_preference FLOAT,      -- Bot detection
    sol_to_token_trades BIGINT,         -- Non-SOL trade analysis
    token_to_sol_trades BIGINT,
    token_to_token_trades BIGINT,
    unique_from_tokens_non_sol BIGINT,
    unique_to_tokens_non_sol BIGINT,
    sol_to_token_percentage FLOAT,
    token_to_sol_percentage FLOAT,
    token_to_token_percentage FLOAT,
    buy_sell_ratio FLOAT,
    total_positions BIGINT,             -- Aggregated performance features  
    win_rate FLOAT,
    avg_pnl_per_position FLOAT,
    avg_roi FLOAT
);
```

### trader_coin_performance Table
```sql
CREATE TABLE trader_coin_performance (
    swapper VARCHAR,                    -- Wallet address
    mint VARCHAR,                       -- Token address
    sol_spent_on_coin DOUBLE,          -- Investment amount
    sol_received_from_coin DOUBLE,     -- Returns
    buy_trades BIGINT,                 -- Trade counts
    sell_trades BIGINT,
    total_coin_trades BIGINT,
    net_sol_pnl_per_coin DOUBLE,       -- Performance metrics
    roi_on_coin DOUBLE,
    hours_active_on_coin DOUBLE        -- Time commitment
);
```

This feature set provides a comprehensive foundation for understanding Solana trader behavior patterns and building predictive models for new coin trading scenarios.