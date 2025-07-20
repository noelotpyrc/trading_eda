-- Aggregate Solana first_day_trades to OHLC format by coin and block timestamp
-- Each row represents one coin at one block timestamp level (similar to NVDA OHLC data)

WITH constants AS (
    SELECT 'So11111111111111111111111111111111111111112' AS SOL_MINT
),

-- Prepare clean trading data with price calculations
clean_trades AS (
    SELECT 
        t.*,
        c.SOL_MINT,
        
        -- Trading direction indicators
        CASE WHEN t.mint = t.swap_to_mint THEN 1 ELSE 0 END AS is_buy,
        CASE WHEN t.mint = t.swap_from_mint THEN 1 ELSE 0 END AS is_sell,
        
        -- Calculate SOL amounts (denominator for price)
        CASE 
            WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_from_amount
            WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_to_amount
            ELSE 0.0
        END AS sol_amount,
        
        -- Calculate token amounts (numerator for price)
        CASE 
            WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_to_amount
            WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_from_amount
            ELSE 0.0
        END AS token_amount
        
    FROM first_day_trades t
    CROSS JOIN constants c
    WHERE 
        -- Filter to successful SOL-related trades only
        t.succeeded = TRUE
        AND (t.swap_from_mint = c.SOL_MINT OR t.swap_to_mint = c.SOL_MINT)
        AND t.mint != c.SOL_MINT
),

-- Calculate prices and filter valid trades
priced_trades AS (
    SELECT 
        *,
        -- Price = SOL amount / token amount (SOL per token)
        sol_amount / NULLIF(token_amount, 0) AS price
        
    FROM clean_trades
    WHERE 
        sol_amount > 0
        AND token_amount > 0
),

-- Add row numbers for OHLC calculation (first/last price in each block)
ordered_trades AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY mint, block_timestamp 
            ORDER BY __row_index ASC
        ) AS trade_order_asc,
        ROW_NUMBER() OVER (
            PARTITION BY mint, block_timestamp 
            ORDER BY __row_index DESC
        ) AS trade_order_desc
        
    FROM priced_trades
),

-- Calculate OHLC aggregations
ohlc_aggregated AS (
    SELECT 
        mint,
        block_timestamp,
        
        -- Time formatting
        STRFTIME(block_timestamp, '%Y%m%d %H:%M:%S') AS time_readable,
        CAST(EPOCH(block_timestamp) * 1000 AS BIGINT) AS timestamp_unix,
        
        -- OHLC prices
        MIN(CASE WHEN trade_order_asc = 1 THEN price END) AS open,
        MAX(price) AS high,
        MIN(price) AS low,
        MIN(CASE WHEN trade_order_desc = 1 THEN price END) AS close,
        
        -- Volume weighted average price (VWAP)
        SUM(price * sol_amount) / NULLIF(SUM(sol_amount), 0) AS vwap,
        
        -- Volume metrics
        SUM(sol_amount) AS sol_volume,
        SUM(token_amount) AS token_volume,
        
        -- Transaction count
        COUNT(*) AS transactions,
        
        -- Price statistics for validation
        COUNT(DISTINCT price) AS unique_prices,
        STDDEV(price) AS price_stddev
        
    FROM ordered_trades
    GROUP BY mint, block_timestamp
    HAVING 
        COUNT(*) >= 1  -- At least 1 transaction per block
        AND SUM(sol_amount) > 0
        AND SUM(token_amount) > 0
),

-- Final formatting to match NVDA structure
final_output AS (
    SELECT 
        time_readable AS time,
        timestamp_unix AS timestamp,
        ROUND(open, 12) AS open,
        ROUND(high, 12) AS high,
        ROUND(low, 12) AS low,
        ROUND(close, 12) AS close,
        ROUND(vwap, 12) AS vwap,
        ROUND(sol_volume, 6) AS sol_volume,
        ROUND(token_volume, 2) AS token_volume,
        transactions,
        mint,
        block_timestamp AS block_timestamp_iso,
        '' AS otc,  -- Empty for compatibility with NVDA format
        
        -- Additional validation columns (can be removed in production)
        unique_prices,
        ROUND(price_stddev, 12) AS price_stddev
        
    FROM ohlc_aggregated
)

SELECT * FROM final_output
ORDER BY mint, block_timestamp_iso;

-- ====================
-- QUERY PERFORMANCE NOTES
-- ====================

/*
For optimal performance on large datasets:

1. **Create indexes**:
   CREATE INDEX idx_trades_mint_timestamp ON first_day_trades(mint, block_timestamp);
   CREATE INDEX idx_trades_sol_mint ON first_day_trades(swap_from_mint, swap_to_mint);
   CREATE INDEX idx_trades_succeeded ON first_day_trades(succeeded);

2. **Partitioning** (if supported):
   Consider partitioning by date if data spans multiple days

3. **Memory considerations**:
   - This query processes all coins at once
   - For very large datasets, consider adding WHERE mint IN (...) to process specific coins
   - Expected reduction: ~1M transactions → ~50K OHLC records

4. **Alternative for specific coins**:
   Add this WHERE clause before ORDER BY:
   WHERE mint IN ('coin1', 'coin2', 'coin3')

5. **Validation queries**:
   -- Check price ranges
   SELECT MIN(open), MAX(high), AVG(vwap) FROM final_output;
   
   -- Check record counts
   SELECT COUNT(*) as total_records, COUNT(DISTINCT mint) as unique_coins FROM final_output;
   
   -- Top coins by volume
   SELECT mint, SUM(sol_volume) as total_volume FROM final_output GROUP BY mint ORDER BY 2 DESC LIMIT 10;
*/

-- ====================
-- EXPECTED OUTPUT FORMAT
-- ====================

/*
Sample output (matches NVDA structure with crypto extensions):

time                  |timestamp    |open      |high      |low       |close     |vwap      |sol_volume|token_volume|transactions|mint           |block_timestamp_iso         |otc|unique_prices|price_stddev
20210410 15:30:45    |1618069845000|0.000123  |0.000125  |0.000121  |0.000124  |0.000123  |1234.56   |9876543.21  |47          |ABC123...DEF789|2021-04-10T15:30:45+00:00 |   |5            |0.000001

Columns explained:
- time: Human readable timestamp (NVDA format)
- timestamp: Unix timestamp in milliseconds  
- open/high/low/close: Price in SOL per token
- vwap: Volume weighted average price
- sol_volume: Total SOL traded in this block
- token_volume: Total tokens traded in this block
- transactions: Number of individual swaps aggregated
- mint: Token contract address (coin identifier)
- block_timestamp_iso: Original ISO timestamp
- otc: Empty (for NVDA compatibility)
- unique_prices: Number of distinct prices (validation)
- price_stddev: Price standard deviation (validation)
*/