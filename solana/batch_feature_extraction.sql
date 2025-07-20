-- Batch Feature Extraction for All Coins - 30/60/120 Second Windows
-- Generates features for all coins at specified sampling intervals
-- Based on batch578_sample_analysis/04-3_ml_ensemble_signals.ipynb

-- ====================================
-- CONFIGURATION PARAMETERS
-- ====================================
-- Adjust these parameters as needed:

WITH config AS (
    SELECT 
        'So11111111111111111111111111111111111111112' AS SOL_MINT,
        60 AS SAMPLING_INTERVAL_SECONDS,  -- Extract features every 60 seconds
        120 AS MIN_LOOKBACK_BUFFER_SECONDS, -- Minimum lookback needed (max of 30,60,120)
        300 AS FORWARD_PREDICTION_WINDOW_SECONDS, -- For future profitability labeling
        100 AS MIN_TRANSACTIONS_PER_COIN -- Skip coins with too little data
),

-- ====================================
-- SAMPLE TIMESTAMP GENERATION
-- ====================================
-- Generate sampling timestamps for each coin
coin_time_ranges AS (
    SELECT 
        c.SOL_MINT,
        c.SAMPLING_INTERVAL_SECONDS,
        c.MIN_LOOKBACK_BUFFER_SECONDS,
        c.FORWARD_PREDICTION_WINDOW_SECONDS,
        c.MIN_TRANSACTIONS_PER_COIN,
        t.mint AS coin_id,
        COUNT(*) AS total_transactions,
        MIN(t.block_timestamp) AS first_timestamp,
        MAX(t.block_timestamp) AS last_timestamp,
        
        -- Calculate valid sampling window (need buffer for lookback and forward)
        MIN(t.block_timestamp) + INTERVAL '1' SECOND * c.MIN_LOOKBACK_BUFFER_SECONDS AS sampling_start,
        MAX(t.block_timestamp) - INTERVAL '1' SECOND * c.FORWARD_PREDICTION_WINDOW_SECONDS AS sampling_end
        
    FROM first_day_trades t
    CROSS JOIN config c
    WHERE 
        -- Filter to SOL-related trades only
        (t.swap_from_mint = c.SOL_MINT OR t.swap_to_mint = c.SOL_MINT)
        AND t.mint != c.SOL_MINT
        AND t.succeeded = TRUE
    GROUP BY c.SOL_MINT, c.SAMPLING_INTERVAL_SECONDS, c.MIN_LOOKBACK_BUFFER_SECONDS, 
             c.FORWARD_PREDICTION_WINDOW_SECONDS, c.MIN_TRANSACTIONS_PER_COIN, t.mint
    HAVING 
        COUNT(*) >= c.MIN_TRANSACTIONS_PER_COIN
        AND MIN(t.block_timestamp) + INTERVAL '1' SECOND * c.MIN_LOOKBACK_BUFFER_SECONDS < 
            MAX(t.block_timestamp) - INTERVAL '1' SECOND * c.FORWARD_PREDICTION_WINDOW_SECONDS
),

-- Generate sample timestamps using generate_series
sampling_timestamps AS (
    SELECT 
        ctr.*,
        gs.sample_timestamp
    FROM coin_time_ranges ctr
    CROSS JOIN LATERAL generate_series(
        ctr.sampling_start,
        ctr.sampling_end,
        INTERVAL '1' SECOND * ctr.SAMPLING_INTERVAL_SECONDS
    ) AS gs(sample_timestamp)
),

-- ====================================
-- PREPARED TRADING DATA
-- ====================================
-- Prepare base trading data with all indicators
prepared_trading_data AS (
    SELECT 
        t.*,
        c.SOL_MINT,
        
        -- Trading direction flags
        CASE WHEN t.mint = t.swap_to_mint THEN 1 ELSE 0 END AS is_buy,
        CASE WHEN t.mint = t.swap_from_mint THEN 1 ELSE 0 END AS is_sell,
        
        -- SOL amount calculation
        CASE 
            WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_from_amount
            WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_to_amount
            ELSE 0.0
        END AS sol_amount,
        
        -- Transaction size categories
        CASE 
            WHEN (
                CASE 
                    WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_from_amount
                    WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_to_amount
                    ELSE 0.0
                END
            ) >= 100 THEN 'Whale'
            WHEN (
                CASE 
                    WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_from_amount
                    WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_to_amount
                    ELSE 0.0
                END
            ) >= 10 THEN 'Big'
            WHEN (
                CASE 
                    WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_from_amount
                    WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_to_amount
                    ELSE 0.0
                END
            ) >= 1 THEN 'Medium'
            WHEN (
                CASE 
                    WHEN t.mint = t.swap_to_mint AND t.swap_from_mint = c.SOL_MINT THEN t.swap_from_amount
                    WHEN t.mint = t.swap_from_mint AND t.swap_to_mint = c.SOL_MINT THEN t.swap_to_amount
                    ELSE 0.0
                END
            ) > 0 THEN 'Small'
            ELSE 'Unknown'
        END AS txn_size_category
        
    FROM first_day_trades t
    CROSS JOIN config c
    WHERE 
        -- Filter to SOL-related trades only
        (t.swap_from_mint = c.SOL_MINT OR t.swap_to_mint = c.SOL_MINT)
        AND t.mint != c.SOL_MINT
        AND t.succeeded = TRUE
),

-- ====================================
-- FEATURE EXTRACTION - ALL TIME WINDOWS
-- ====================================
-- Extract features for 30s, 60s, and 120s windows in one pass
extracted_features AS (
    SELECT 
        st.coin_id,
        st.sample_timestamp,
        st.total_transactions,
        
        -- ====================
        -- 30-SECOND FEATURES
        -- ====================
        
        -- Volume features (30s)
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS total_volume_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.is_buy = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS buy_volume_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.is_sell = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS sell_volume_30s,
        
        -- Transaction flow features (30s)
        COALESCE(COUNT(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                            AND ptd.block_timestamp < st.sample_timestamp 
                            THEN 1 END), 0) AS total_txns_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.is_buy ELSE 0 END), 0) AS buy_txns_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.is_sell ELSE 0 END), 0) AS sell_txns_30s,
        
        -- Trader behavior features (30s)
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     THEN ptd.swapper END), 0) AS unique_traders_30s,
        
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     AND ptd.is_buy = 1 
                                     THEN ptd.swapper END), 0) AS unique_buyers_30s,
        
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     AND ptd.is_sell = 1 
                                     THEN ptd.swapper END), 0) AS unique_sellers_30s,
        
        -- Transaction size distribution (30s)
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Small' 
                          THEN 1 ELSE 0 END), 0) AS small_txns_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Medium' 
                          THEN 1 ELSE 0 END), 0) AS medium_txns_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Big' 
                          THEN 1 ELSE 0 END), 0) AS big_txns_30s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Whale' 
                          THEN 1 ELSE 0 END), 0) AS whale_txns_30s,
        
        -- Volume concentration (30s)
        COALESCE(STDDEV_POP(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                                 AND ptd.block_timestamp < st.sample_timestamp 
                                 THEN ptd.sol_amount END) / NULLIF(AVG(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                                                                           AND ptd.block_timestamp < st.sample_timestamp 
                                                                           THEN ptd.sol_amount END) + 1e-10, 0), 0.0) AS volume_concentration_30s,
        
        -- Volume concentration breakdown (30s)
        COALESCE(AVG(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.sol_amount END), 0.0) AS volume_mean_30s,
        COALESCE(STDDEV_POP(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '30 seconds' 
                                  AND ptd.block_timestamp < st.sample_timestamp 
                                  THEN ptd.sol_amount END), 0.0) AS volume_std_30s,
        
        -- ====================
        -- 60-SECOND FEATURES
        -- ====================
        
        -- Volume features (60s)
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS total_volume_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.is_buy = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS buy_volume_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.is_sell = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS sell_volume_60s,
        
        -- Transaction flow features (60s)
        COALESCE(COUNT(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                            AND ptd.block_timestamp < st.sample_timestamp 
                            THEN 1 END), 0) AS total_txns_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.is_buy ELSE 0 END), 0) AS buy_txns_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.is_sell ELSE 0 END), 0) AS sell_txns_60s,
        
        -- Trader behavior features (60s)
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     THEN ptd.swapper END), 0) AS unique_traders_60s,
        
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     AND ptd.is_buy = 1 
                                     THEN ptd.swapper END), 0) AS unique_buyers_60s,
        
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     AND ptd.is_sell = 1 
                                     THEN ptd.swapper END), 0) AS unique_sellers_60s,
        
        -- Transaction size distribution (60s)
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Small' 
                          THEN 1 ELSE 0 END), 0) AS small_txns_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Medium' 
                          THEN 1 ELSE 0 END), 0) AS medium_txns_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Big' 
                          THEN 1 ELSE 0 END), 0) AS big_txns_60s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Whale' 
                          THEN 1 ELSE 0 END), 0) AS whale_txns_60s,
        
        -- Volume concentration (60s)
        COALESCE(STDDEV_POP(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                                 AND ptd.block_timestamp < st.sample_timestamp 
                                 THEN ptd.sol_amount END) / NULLIF(AVG(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                                                                           AND ptd.block_timestamp < st.sample_timestamp 
                                                                           THEN ptd.sol_amount END), 0), 0.0) AS volume_concentration_60s,
        
        -- Volume concentration breakdown (60s)
        COALESCE(AVG(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.sol_amount END), 0.0) AS volume_mean_60s,
        COALESCE(STDDEV_POP(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '60 seconds' 
                                  AND ptd.block_timestamp < st.sample_timestamp 
                                  THEN ptd.sol_amount END), 0.0) AS volume_std_60s,
        
        -- ====================
        -- 120-SECOND FEATURES
        -- ====================
        
        -- Volume features (120s)
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS total_volume_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.is_buy = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS buy_volume_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.is_sell = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS sell_volume_120s,
        
        -- Transaction flow features (120s)
        COALESCE(COUNT(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                            AND ptd.block_timestamp < st.sample_timestamp 
                            THEN 1 END), 0) AS total_txns_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.is_buy ELSE 0 END), 0) AS buy_txns_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.is_sell ELSE 0 END), 0) AS sell_txns_120s,
        
        -- Trader behavior features (120s)
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     THEN ptd.swapper END), 0) AS unique_traders_120s,
        
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     AND ptd.is_buy = 1 
                                     THEN ptd.swapper END), 0) AS unique_buyers_120s,
        
        COALESCE(COUNT(DISTINCT CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                                     AND ptd.block_timestamp < st.sample_timestamp 
                                     AND ptd.is_sell = 1 
                                     THEN ptd.swapper END), 0) AS unique_sellers_120s,
        
        -- Transaction size distribution (120s)
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Small' 
                          THEN 1 ELSE 0 END), 0) AS small_txns_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Medium' 
                          THEN 1 ELSE 0 END), 0) AS medium_txns_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Big' 
                          THEN 1 ELSE 0 END), 0) AS big_txns_120s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          AND ptd.txn_size_category = 'Whale' 
                          THEN 1 ELSE 0 END), 0) AS whale_txns_120s,
        
        -- Volume concentration (120s)
        COALESCE(STDDEV_POP(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                                 AND ptd.block_timestamp < st.sample_timestamp 
                                 THEN ptd.sol_amount END) / NULLIF(AVG(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                                                                           AND ptd.block_timestamp < st.sample_timestamp 
                                                                           THEN ptd.sol_amount END), 0), 0.0) AS volume_concentration_120s,
        
        -- Volume concentration breakdown (120s)
        COALESCE(AVG(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                          AND ptd.block_timestamp < st.sample_timestamp 
                          THEN ptd.sol_amount END), 0.0) AS volume_mean_120s,
        COALESCE(STDDEV_POP(CASE WHEN ptd.block_timestamp >= st.sample_timestamp - INTERVAL '120 seconds' 
                                  AND ptd.block_timestamp < st.sample_timestamp 
                                  THEN ptd.sol_amount END), 0.0) AS volume_std_120s,
        
        -- ====================
        -- FORWARD PROFITABILITY LABEL (OPTIONAL)
        -- ====================
        
        -- Calculate buy vs sell pressure in the next 300 seconds for labeling
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp 
                          AND ptd.block_timestamp < st.sample_timestamp + INTERVAL '300 seconds'
                          AND ptd.is_buy = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS forward_buy_volume_300s,
        
        COALESCE(SUM(CASE WHEN ptd.block_timestamp >= st.sample_timestamp 
                          AND ptd.block_timestamp < st.sample_timestamp + INTERVAL '300 seconds'
                          AND ptd.is_sell = 1 
                          THEN ptd.sol_amount ELSE 0 END), 0.0) AS forward_sell_volume_300s
        
    FROM sampling_timestamps st
    LEFT JOIN prepared_trading_data ptd ON ptd.mint = st.coin_id
    GROUP BY st.coin_id, st.sample_timestamp, st.total_transactions
),

-- ====================================
-- FINAL FEATURE CALCULATION
-- ====================================
-- Calculate derived features (ratios, imbalances, etc.)
final_features AS (
    SELECT 
        coin_id,
        sample_timestamp,
        total_transactions,
        
        -- ====================
        -- 30-SECOND DERIVED FEATURES
        -- ====================
        
        -- Volume features
        total_volume_30s,
        buy_volume_30s,
        sell_volume_30s,
        CASE WHEN total_volume_30s > 0 THEN buy_volume_30s / total_volume_30s ELSE 0.0 END AS buy_ratio_30s,
        CASE WHEN total_volume_30s > 0 THEN (buy_volume_30s - sell_volume_30s) / total_volume_30s ELSE 0.0 END AS volume_imbalance_30s,
        
        -- Transaction flow features
        total_txns_30s,
        buy_txns_30s,
        sell_txns_30s,
        CASE WHEN total_txns_30s > 0 THEN buy_txns_30s::FLOAT / total_txns_30s ELSE 0.0 END AS txn_buy_ratio_30s,
        CASE WHEN total_txns_30s > 0 THEN (buy_txns_30s - sell_txns_30s)::FLOAT / total_txns_30s ELSE 0.0 END AS txn_flow_imbalance_30s,
        
        -- Trader behavior features
        unique_traders_30s,
        unique_buyers_30s,
        unique_sellers_30s,
        CASE WHEN unique_traders_30s > 0 THEN unique_buyers_30s::FLOAT / unique_traders_30s ELSE 0.0 END AS trader_buy_ratio_30s,
        
        -- Transaction size features
        CASE WHEN total_txns_30s > 0 AND total_volume_30s > 0 THEN total_volume_30s / total_txns_30s ELSE 0.0 END AS avg_txn_size_30s,
        CASE WHEN unique_traders_30s > 0 THEN total_volume_30s / unique_traders_30s ELSE 0.0 END AS volume_per_trader_30s,
        volume_concentration_30s,
        volume_mean_30s,
        volume_std_30s,
        CASE WHEN total_txns_30s > 0 THEN small_txns_30s::FLOAT / total_txns_30s ELSE 0.0 END AS small_txn_ratio_30s,
        CASE WHEN total_txns_30s > 0 THEN medium_txns_30s::FLOAT / total_txns_30s ELSE 0.0 END AS medium_txn_ratio_30s,
        CASE WHEN total_txns_30s > 0 THEN big_txns_30s::FLOAT / total_txns_30s ELSE 0.0 END AS big_txn_ratio_30s,
        CASE WHEN total_txns_30s > 0 THEN whale_txns_30s::FLOAT / total_txns_30s ELSE 0.0 END AS whale_txn_ratio_30s,
        
        -- ====================
        -- 60-SECOND DERIVED FEATURES
        -- ====================
        
        -- Volume features
        total_volume_60s,
        buy_volume_60s,
        sell_volume_60s,
        CASE WHEN total_volume_60s > 0 THEN buy_volume_60s / total_volume_60s ELSE 0.0 END AS buy_ratio_60s,
        CASE WHEN total_volume_60s > 0 THEN (buy_volume_60s - sell_volume_60s) / total_volume_60s ELSE 0.0 END AS volume_imbalance_60s,
        
        -- Transaction flow features
        total_txns_60s,
        buy_txns_60s,
        sell_txns_60s,
        CASE WHEN total_txns_60s > 0 THEN buy_txns_60s::FLOAT / total_txns_60s ELSE 0.0 END AS txn_buy_ratio_60s,
        CASE WHEN total_txns_60s > 0 THEN (buy_txns_60s - sell_txns_60s)::FLOAT / total_txns_60s ELSE 0.0 END AS txn_flow_imbalance_60s,
        
        -- Trader behavior features
        unique_traders_60s,
        unique_buyers_60s,
        unique_sellers_60s,
        CASE WHEN unique_traders_60s > 0 THEN unique_buyers_60s::FLOAT / unique_traders_60s ELSE 0.0 END AS trader_buy_ratio_60s,
        
        -- Transaction size features
        CASE WHEN total_txns_60s > 0 AND total_volume_60s > 0 THEN total_volume_60s / total_txns_60s ELSE 0.0 END AS avg_txn_size_60s,
        CASE WHEN unique_traders_60s > 0 THEN total_volume_60s / unique_traders_60s ELSE 0.0 END AS volume_per_trader_60s,
        volume_concentration_60s,
        volume_mean_60s,
        volume_std_60s,
        CASE WHEN total_txns_60s > 0 THEN small_txns_60s::FLOAT / total_txns_60s ELSE 0.0 END AS small_txn_ratio_60s,
        CASE WHEN total_txns_60s > 0 THEN medium_txns_60s::FLOAT / total_txns_60s ELSE 0.0 END AS medium_txn_ratio_60s,
        CASE WHEN total_txns_60s > 0 THEN big_txns_60s::FLOAT / total_txns_60s ELSE 0.0 END AS big_txn_ratio_60s,
        CASE WHEN total_txns_60s > 0 THEN whale_txns_60s::FLOAT / total_txns_60s ELSE 0.0 END AS whale_txn_ratio_60s,
        
        -- ====================
        -- 120-SECOND DERIVED FEATURES
        -- ====================
        
        -- Volume features
        total_volume_120s,
        buy_volume_120s,
        sell_volume_120s,
        CASE WHEN total_volume_120s > 0 THEN buy_volume_120s / total_volume_120s ELSE 0.0 END AS buy_ratio_120s,
        CASE WHEN total_volume_120s > 0 THEN (buy_volume_120s - sell_volume_120s) / total_volume_120s ELSE 0.0 END AS volume_imbalance_120s,
        
        -- Transaction flow features
        total_txns_120s,
        buy_txns_120s,
        sell_txns_120s,
        CASE WHEN total_txns_120s > 0 THEN buy_txns_120s::FLOAT / total_txns_120s ELSE 0.0 END AS txn_buy_ratio_120s,
        CASE WHEN total_txns_120s > 0 THEN (buy_txns_120s - sell_txns_120s)::FLOAT / total_txns_120s ELSE 0.0 END AS txn_flow_imbalance_120s,
        
        -- Trader behavior features
        unique_traders_120s,
        unique_buyers_120s,
        unique_sellers_120s,
        CASE WHEN unique_traders_120s > 0 THEN unique_buyers_120s::FLOAT / unique_traders_120s ELSE 0.0 END AS trader_buy_ratio_120s,
        
        -- Transaction size features
        CASE WHEN total_txns_120s > 0 AND total_volume_120s > 0 THEN total_volume_120s / total_txns_120s ELSE 0.0 END AS avg_txn_size_120s,
        CASE WHEN unique_traders_120s > 0 THEN total_volume_120s / unique_traders_120s ELSE 0.0 END AS volume_per_trader_120s,
        volume_concentration_120s,
        volume_mean_120s,
        volume_std_120s,
        CASE WHEN total_txns_120s > 0 THEN small_txns_120s::FLOAT / total_txns_120s ELSE 0.0 END AS small_txn_ratio_120s,
        CASE WHEN total_txns_120s > 0 THEN medium_txns_120s::FLOAT / total_txns_120s ELSE 0.0 END AS medium_txn_ratio_120s,
        CASE WHEN total_txns_120s > 0 THEN big_txns_120s::FLOAT / total_txns_120s ELSE 0.0 END AS big_txn_ratio_120s,
        CASE WHEN total_txns_120s > 0 THEN whale_txns_120s::FLOAT / total_txns_120s ELSE 0.0 END AS whale_txn_ratio_120s,
        
        -- ====================
        -- PROFITABILITY LABEL
        -- ====================
        
        forward_buy_volume_300s,
        forward_sell_volume_300s,
        CASE WHEN forward_buy_volume_300s > forward_sell_volume_300s THEN 1 ELSE 0 END AS is_profitable_300s
        
    FROM extracted_features
)

SELECT * FROM final_features
ORDER BY coin_id, sample_timestamp;

-- ====================
-- USAGE INSTRUCTIONS
-- ====================

/*
This query will:

1. **Automatically sample all coins** at 60-second intervals
2. **Extract 60 features** across 30s/60s/120s windows for each sample
3. **Generate profitability labels** for the next 300 seconds
4. **Filter out low-activity coins** (< 100 transactions)
5. **Handle edge cases** with proper time buffers

Expected output columns (63 total):
- coin_id, sample_timestamp, total_transactions
- 21 features × 3 time windows (30s, 60s, 120s) including volume_per_trader
- Forward profitability labels

To optimize performance:
1. **Create indexes** on mint, block_timestamp, swapper
2. **Partition by date** if your data spans multiple days
3. **Adjust sampling interval** (currently 60 seconds) based on needs
4. **Filter specific coins** by adding WHERE coin_id IN (...) if needed

To add volume_concentration features:
- Calculate standard deviation within each window
- This requires window functions or separate aggregations

REGIME DETECTION USAGE:
The volume_per_trader features enable regime detection for whale activity periods:

1. **Normal Trading Regime**: volume_per_trader ~3-7 SOL per trader
2. **Whale Activity Regime (Regime 2)**: volume_per_trader 28x higher (~87-154 SOL)
3. **Detection Logic**:
   ```sql
   -- Identify potential Regime 2 (Whale Activity) periods
   SELECT *, 
     CASE 
       WHEN volume_per_trader_60s > 50 AND 
            whale_txn_ratio_30s > 0.002 AND 
            avg_txn_size_60s > 4.0 
       THEN 'REGIME_2_WHALE_ACTIVITY'
       ELSE 'NORMAL_TRADING' 
     END AS market_regime
   FROM final_features
   ```

4. **Expected Performance Boost**:
   - General ML model: AUC ~0.70
   - Regime 2 periods: AUC ~0.80+ (+15.3% improvement)
   - Occurs ~11.5% of time (selective trading opportunity)

5. **Trading Strategy**:
   - Wait for volume_per_trader_60s > 50 (whale activity detection)
   - Apply specialized Regime 2 model during these periods
   - Use 2x position sizing when in high-signal regime
   - Default to conservative trading in normal periods

This single query replaces thousands of individual coin queries AND provides regime detection capability!
*/