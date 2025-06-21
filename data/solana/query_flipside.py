# 1) pip install flipside                         # one-time
from flipside import Flipside
import pandas as pd
import dotenv
import os
import time
from datetime import datetime
import json
import argparse
env = dotenv.load_dotenv()

# --- config -------------------------------------------------
API_KEY      = os.getenv("FLIPSIDE_API_KEY")
# MINT_ADDRESS = "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm"  # commented out, not needed for batch

flipside = Flipside(API_KEY, "https://api-v2.flipsidecrypto.xyz")

# SQL template - we'll replace {{MINT_ADDRESS}} with the actual value
sql_template = """
/*  ─────────────────  PARAMETER  ─────────────────
    MINT_LIST   ← stringified JSON array
                      '[ "mint1", "mint2", … ]'
   ─────────────────────────────────────────────── */

/* 0️⃣  turn the JSON list into a row-set */
WITH mint_set AS (
    SELECT value::string AS mint
    FROM   TABLE(FLATTEN( INPUT => PARSE_JSON('{{MINT_LIST}}') ))
),

/* 1️⃣  earliest timestamp *per mint* on either side of a swap               */
first_trade AS (
    SELECT
        mint,
        MIN(block_timestamp) AS first_ts
    FROM (

        /* main swaps table, both legs */
        SELECT ms.mint, fs.block_timestamp
        FROM   solana.defi.fact_swaps            fs
        JOIN   mint_set                          ms
               ON  fs.SWAP_FROM_MINT = ms.mint
                OR fs.SWAP_TO_MINT   = ms.mint

        UNION ALL

        /* Jupiter-only splits (post-25-Jul-2024), both legs */
        SELECT ms.mint, js.block_timestamp
        FROM   solana.defi.fact_swaps_jupiter_inner  js
        JOIN   mint_set                              ms
               ON  js.SWAP_FROM_MINT = ms.mint
                OR js.SWAP_TO_MINT   = ms.mint
    )
    GROUP BY mint
),

/* 2️⃣  pre-compute each mint's [day_start, day_end) window                   */
day_window AS (
    SELECT
        mint,
        DATE_TRUNC('day', first_ts)                    AS day_start,
        DATE_TRUNC('day', first_ts) + INTERVAL '1 day' AS day_end
    FROM first_trade
),

/* 3️⃣  pull swaps for those mints only once (union of the two sources)       */
all_swaps AS (
    SELECT
        BLOCK_TIMESTAMP, BLOCK_ID, TX_ID, SUCCEEDED, SWAPPER,
        SWAP_FROM_AMOUNT, SWAP_FROM_MINT,
        SWAP_TO_AMOUNT,   SWAP_TO_MINT,
        INSERTED_TIMESTAMP, MODIFIED_TIMESTAMP
    FROM   solana.defi.fact_swaps
    WHERE  SWAP_FROM_MINT IN (SELECT mint FROM mint_set)
       OR  SWAP_TO_MINT   IN (SELECT mint FROM mint_set)

    UNION ALL
    SELECT
        BLOCK_TIMESTAMP, BLOCK_ID, TX_ID, SUCCEEDED, SWAPPER,
        SWAP_FROM_AMOUNT, SWAP_FROM_MINT,
        SWAP_TO_AMOUNT,   SWAP_TO_MINT,
        INSERTED_TIMESTAMP, MODIFIED_TIMESTAMP
    FROM   solana.defi.fact_swaps_jupiter_inner
    WHERE  SWAP_FROM_MINT IN (SELECT mint FROM mint_set)
       OR  SWAP_TO_MINT   IN (SELECT mint FROM mint_set)
)

/* 4️⃣  final filter: keep only trades inside each mint's first-day window    */
SELECT
    w.mint,
    s.BLOCK_TIMESTAMP,
    s.SUCCEEDED,
    s.SWAPPER,
    s.SWAP_FROM_AMOUNT,
    s.SWAP_FROM_MINT,
    s.SWAP_TO_AMOUNT,
    s.SWAP_TO_MINT
FROM        all_swaps  AS s
INNER JOIN  day_window AS w
       ON  (   s.SWAP_FROM_MINT = w.mint
            OR s.SWAP_TO_MINT   = w.mint )
       AND  s.BLOCK_TIMESTAMP >= w.day_start
       AND  s.BLOCK_TIMESTAMP <  w.day_end
ORDER BY w.mint, s.BLOCK_TIMESTAMP;
"""

# --- Commented out: single-mint query code ---
# sql = sql_template.replace('{{MINT_ADDRESS}}', MINT_ADDRESS)
# print(f"Querying data for mint address: {MINT_ADDRESS}")
# try:
#     # 2) kick off the query with page 1 / size 100
#     query_run = flipside.query(
#         sql,
#         page_number=1,
#         page_size=1  # Increased from 1 for efficiency
#     )
#     print(query_run)
#     # Check if query was successful
#     if not query_run or not hasattr(query_run, 'query_id'):
#         print("Query failed to initialize")
#         exit(1)
#     # 3) stream all pages
#     all_rows = []
#     current_page = 1
#     total_pages = 1  # Start with assumption of at least 1 page
#     while current_page <= total_pages:
#         page = flipside.get_query_results(
#             query_run.query_id,
#             page_number=current_page,
#             page_size=100000
#         )
#         if hasattr(page, 'page') and hasattr(page.page, 'totalPages'):
#             total_pages = page.page.totalPages
#         print(f"Page {current_page} of {total_pages}")
#         if hasattr(page, 'records'):
#             all_rows.extend(page.records or [])
#         current_page += 1
#     # 4) turn into a DataFrame, save, etc.
#     df = pd.DataFrame(all_rows)
#     if len(df) > 0:
#         df.to_csv("first_day_trades.csv")
#         print(f"Fetched {len(df):,} rows → first_day_trades.csv")
#     else:
#         print(f"No trades found for mint address: {MINT_ADDRESS}")
# except Exception as e:
#     print(f"Error: {e}")
# --- End commented out ---

MINTS_CSV = "solana_low_fdv_mints.csv"
OUTPUT_DIR = "first_day_trades/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read all mint addresses
mints = pd.read_csv(MINTS_CSV)["mint_address"].tolist()

BATCH_SIZE = 10

parser = argparse.ArgumentParser(description="Batch Flipside query for mint addresses.")
parser.add_argument('--start-batch', type=int, default=0, help='Start batch index (inclusive)')
parser.add_argument('--end-batch', type=int, default=None, help='End batch index (exclusive)')
args = parser.parse_args()

num_batches = (len(mints) + BATCH_SIZE - 1) // BATCH_SIZE
start_batch = args.start_batch
end_batch = args.end_batch if args.end_batch is not None else num_batches

for batch_idx in range(start_batch, end_batch):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(mints))
    batch = mints[batch_start:batch_end]
    if not batch:
        break
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Batch {batch_idx+1}: {now} Querying data for {len(batch)} mint addresses")
    mint_list_sql = json.dumps(batch)
    sql = sql_template.replace("{{MINT_LIST}}", mint_list_sql)
    try:
        query_run = flipside.query(
            sql,
            page_number=1,
            page_size=1
        )
        if not query_run or not hasattr(query_run, 'query_id'):
            print(f"Query failed to initialize for batch {batch_idx+1}")
            continue
        all_rows = []
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            page = flipside.get_query_results(
                query_run.query_id,
                page_number=current_page,
                page_size=100000
            )
            if hasattr(page, 'page') and hasattr(page.page, 'totalPages'):
                total_pages = page.page.totalPages
            if hasattr(page, 'records'):
                all_rows.extend(page.records or [])
            current_page += 1
        df = pd.DataFrame(all_rows)
        if len(df) > 0:
            out_path = os.path.join(OUTPUT_DIR, f"first_day_trades_batch_{batch_idx+1}.csv")
            df.to_csv(out_path, index=False)
            print(f"Fetched {len(df):,} rows → {out_path}")
        else:
            print(f"No trades found for batch {batch_idx+1}")
    except Exception as e:
        print(f"Error for batch {batch_idx+1}: {e}")
    time.sleep(2)  # polite delay to avoid rate limits