# For pulling the historical trades

Findings so far:

- Polymarket official API has rate limit and /trades has limit on number of trades to return (500)
- All trades are recorded onchain, so Dune or Flipside also has this data

Current plan:

- Use /events API to get all the resolved events
- Construct a query with condition_id (from events) to download the trades data from Dune
- Use /price-history API to get historical price data (for sanity check, and it seems fidelity should be larger than 720min)

Question to answer:

- Need to understand how orders are matched and filled