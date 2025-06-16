# 1) pip install flipside                         # one-time
from flipside import Flipside
import pandas as pd
import dotenv
import os
env = dotenv.load_dotenv()

# --- config -------------------------------------------------
API_KEY      = os.getenv("FLIPSIDE_API_KEY")
MINT_ADDRESS = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"

flipside = Flipside(API_KEY, "https://api-v2.flipsidecrypto.xyz")

# SQL template - we'll replace {{MINT_ADDRESS}} with the actual value
sql_template = """
WITH first_trade AS (
    SELECT MIN(block_timestamp) AS first_ts
    FROM (
        SELECT block_timestamp
        FROM   solana.defi.fact_swaps
        WHERE  SWAP_FROM_MINT = '{{MINT_ADDRESS}}'
           OR  SWAP_TO_MINT   = '{{MINT_ADDRESS}}'
        UNION ALL
        SELECT block_timestamp
        FROM   solana.defi.fact_swaps_jupiter_inner
        WHERE  SWAP_FROM_MINT = '{{MINT_ADDRESS}}'
           OR  SWAP_TO_MINT   = '{{MINT_ADDRESS}}'
    )
),
day_window AS (
    SELECT  DATE_TRUNC('day', first_ts)                    AS day_start,
            DATE_TRUNC('day', first_ts) + INTERVAL '1 day' AS day_end
    FROM    first_trade
),
all_swaps AS (
    SELECT
        BLOCK_TIMESTAMP, BLOCK_ID, TX_ID, SUCCEEDED, SWAPPER,
        SWAP_FROM_AMOUNT, SWAP_FROM_MINT,
        SWAP_TO_AMOUNT,   SWAP_TO_MINT,
        INSERTED_TIMESTAMP, MODIFIED_TIMESTAMP
    FROM   solana.defi.fact_swaps
    WHERE  SWAP_FROM_MINT = '{{MINT_ADDRESS}}'
       OR  SWAP_TO_MINT   = '{{MINT_ADDRESS}}'
    UNION ALL
    SELECT
        BLOCK_TIMESTAMP, BLOCK_ID, TX_ID, SUCCEEDED, SWAPPER,
        SWAP_FROM_AMOUNT, SWAP_FROM_MINT,
        SWAP_TO_AMOUNT,   SWAP_TO_MINT,
        INSERTED_TIMESTAMP, MODIFIED_TIMESTAMP
    FROM   solana.defi.fact_swaps_jupiter_inner
    WHERE  SWAP_FROM_MINT = '{{MINT_ADDRESS}}'
       OR  SWAP_TO_MINT   = '{{MINT_ADDRESS}}'
)
SELECT *
FROM   all_swaps, day_window
WHERE  BLOCK_TIMESTAMP >= day_window.day_start
  AND  BLOCK_TIMESTAMP <  day_window.day_end
ORDER  BY BLOCK_TIMESTAMP;
"""

# Replace the placeholder with the actual mint address
sql = sql_template.replace('{{MINT_ADDRESS}}', MINT_ADDRESS)

print(f"Querying data for mint address: {MINT_ADDRESS}")

try:
    # 2) kick off the query with page 1 / size 100
    query_run = flipside.query(
        sql,
        page_number=1,
        page_size=1  # Increased from 1 for efficiency
    )
    print(query_run)
    # Check if query was successful
    if not query_run or not hasattr(query_run, 'query_id'):
        print("Query failed to initialize")
        exit(1)
    
    # 3) stream all pages
    all_rows = []
    current_page = 1
    total_pages = 1  # Start with assumption of at least 1 page
    
    while current_page <= total_pages:
        page = flipside.get_query_results(
            query_run.query_id,
            page_number=current_page,
            page_size=100000
        )
        
        if hasattr(page, 'page') and hasattr(page.page, 'totalPages'):
            total_pages = page.page.totalPages
        
        print(f"Page {current_page} of {total_pages}")
        
        if hasattr(page, 'records'):
            all_rows.extend(page.records or [])
        
        current_page += 1
    
    # 4) turn into a DataFrame, save, etc.
    df = pd.DataFrame(all_rows)
    
    if len(df) > 0:
        df.to_csv("first_day_trades.csv")
        print(f"Fetched {len(df):,} rows → first_day_trades.csv")
    else:
        print(f"No trades found for mint address: {MINT_ADDRESS}")

except Exception as e:
    print(f"Error: {e}")