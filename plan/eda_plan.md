# Exploratory Data Analysis (EDA) Plan for NVDA Minute-Level OHLCV Data

## 1. Data Loading & Cleaning
- Load all CSVs, combine into a single DataFrame, and add a `date` column from filenames.
- Check and enforce correct data types for all columns.
- Remove or investigate duplicate rows.

## 2. Data Quality Checks
- Check for missing values in all columns.
- Identify and handle zero or negative values in price/volume columns.
- Investigate and document any gaps in the time series (missing minutes/days).

## 3. Time Coverage & Granularity
- Summarize the date range covered by the data.
- Count the number of trading days and minutes per day.
- Visualize the distribution of records per day and identify any days with missing or extra data.
- Segment data by regular and extended trading hours.

## 4. Descriptive Statistics
- Compute summary statistics (mean, std, min, max, percentiles) for all numeric columns (open, high, low, close, vwap, volume, transactions).
- Plot histograms or KDE plots for price and volume columns.

## 5. Time Series Visualizations
- Plot close price and volume over time for the full dataset and for selected sample days/weeks.
- Create candlestick charts for selected days.

## 6. Intraday Patterns
- Calculate and plot the average intraday profile for price and volume (mean per minute across all days).
- Analyze volatility by time of day (standard deviation of returns per minute across days).

## 7. Price Dynamics & Volatility
- Compute and plot minute-to-minute returns and their distribution.
- Calculate rolling volatility (e.g., 30-min window) and plot over time.
- Plot autocorrelation of returns and volume.

## 8. Outlier & Anomaly Detection
- Identify and visualize minutes with extreme price or volume changes.
- Detect and investigate price jumps or drops exceeding a set threshold.

## 9. Market Microstructure Analysis
- Compare VWAP to close price and analyze their relationship.
- Analyze the number of transactions per minute and its distribution.

## 10. Calendar Effects
- Analyze patterns by day of week and by month (e.g., average volume or volatility).
- Visualize and summarize any calendar-based trends.

## 11. Summary of Insights
- Document key findings, patterns, and any data issues discovered during EDA.
- Suggest next steps for further analysis or modeling.

## 12. Microstructure Mean Reversion Analysis (HFT Focus)
- Analyze 1-minute, 5-minute, and 15-minute mean reversion patterns.
- Detect bid-ask bounce patterns using OHLC spread proxies.
- Identify overreaction/correction cycles after large price moves.
- Calculate half-life of price dislocations and optimal entry/exit timing.

## 13. Volume-Price Relationship & Order Flow
- Analyze volume spikes vs. price movement correlations.
- Identify volume exhaustion signals (high volume with minimal price movement).
- Detect volume confirmation patterns (price and volume alignment).
- Create statistical models for unusual volume detection and anomalies.

## 14. Volatility Forecasting Models
- Implement GARCH modeling on minute-level returns for volatility prediction.
- Analyze realized volatility patterns and clustering persistence.
- Develop volatility breakout detection algorithms.
- Create volatility regime switching models for strategy adaptation.

## 15. Intraday Momentum & Reversal Signals
- Analyze 1-5 minute momentum persistence patterns.
- Identify momentum exhaustion signals and reversal indicators.
- Detect reversal patterns after extreme moves (>2-3 standard deviations).
- Analyze price acceleration and deceleration patterns.

## 16. Price Level Clustering & Support/Resistance
- Analyze round number effects and psychological price levels.
- Validate support and resistance level effectiveness.
- Study price magnetic effects near key technical levels.
- Calculate breakout and breakdown probabilities at significant levels.

## 17. Gap Analysis & Trading Opportunities
- Analyze overnight gaps (close-to-open) and their fill probabilities.
- Detect intraday gaps (minute-to-minute jumps) and patterns.
- Study gap fill timing and probability by gap size.
- Correlate gap characteristics with fill speed and success rates.

## 18. Cross-Timeframe Signal Analysis
- Compare momentum signals across 1-minute, 5-minute, and 15-minute timeframes.
- Identify timeframe divergence signals and their predictive power.
- Analyze cascade effects (longer timeframe trends → shorter timeframe entries).
- Optimize signal strength by timeframe combination strategies.

## 19. Liquidity Patterns & Market Impact
- Create effective spread proxies using High-Low and Close-to-Close methods.
- Identify liquidity dry-ups (periods of wide spreads and low volume).
- Analyze optimal execution timing to minimize market impact.
- Model transaction costs and their impact on strategy profitability.

## 20. Correlation & Relative Value Analysis
- Analyze NVDA correlation stability with QQQ, SMH, and other tech stocks.
- Identify correlation breakdown periods and arbitrage opportunities.
- Study sector rotation effects and their impact on relative performance.
- Develop pairs trading signals based on correlation divergences.

## 21. Technical Indicator Optimization
- Optimize RSI parameters for 1-30 minute lookback periods.
- Test MACD parameter combinations for optimal signal generation.
- Analyze Bollinger Band effectiveness across different timeframes.
- Develop custom oscillators tailored to NVDA's specific behavior patterns.

## 22. Pattern Recognition & Seasonality
- Identify recurring micro-patterns (flags, pennants, triangles) in 1-15 minute data.
- Analyze day-of-week effects combined with hour-of-day patterns.
- Study month-end, quarter-end, and earnings season effects.
- Detect holiday adjacency effects and their trading implications.

## 23. Strategy Backtesting Framework Setup
- Develop mean reversion strategy parameters and optimization.
- Create breakout strategy validation with volume confirmation.
- Build scalping opportunity detection for rapid profit targets.
- Design position sizing algorithms based on volatility forecasting.

## 24. Risk Management & Performance Metrics
- Calculate maximum drawdown periods and recovery times.
- Analyze trade win/loss ratios by strategy type and market conditions.
- Develop dynamic stop-loss placement based on volatility regimes.
- Create performance attribution analysis by time-of-day and market conditions. 