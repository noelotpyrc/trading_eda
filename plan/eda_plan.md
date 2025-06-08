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