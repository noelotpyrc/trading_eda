## Polymarket Gamma Events Data Dictionary (based on saved JSON)

This reflects the actual keys present in `data/polymarket/gamma_events_test_jan2024.json` (149 events, Jan 1–Feb 1, 2024).

### Dataset
- JSON array of Event objects.
- Each Event includes core metadata and an embedded `markets` array (list of Market objects).

### Event object (observed keys)
- id (string)
- ticker (string)
- slug (string)
- title (string)
- description (string)
- resolutionSource (string)
- startDate (string, ISO datetime)
- creationDate (string, ISO datetime)
- endDate (string, ISO datetime)
- image (string URL)
- icon (string URL)
- active (boolean)
- closed (boolean)
- archived (boolean)
- new (boolean)
- featured (boolean)
- restricted (boolean)
- liquidity (number | string)
- volume (number | string)
- openInterest (number | string)
- sortBy (string)
- published_at (string)
- createdAt (string)
- updatedAt (string)
- commentsEnabled (boolean)
- competitive (number)
- volume24hr, volume1wk, volume1mo, volume1yr (number)
- enableOrderBook (boolean)
- liquidityAmm, liquidityClob (number)
- commentCount (number)
- markets (array[Market])
- tags (array[Tag])
- cyom (boolean)
- closedTime (string)
- showAllOutcomes (boolean)
- showMarketImages (boolean)
- enableNegRisk (boolean)
- negRiskAugmented (boolean)
- pendingDeployment (boolean)
- deploying (boolean)
- updatedBy (number | string | null)
- automaticallyResolved (boolean | null)
- automaticallyActive (boolean | null)
- featuredImage (string | null)
- negRisk (boolean | null)
- negRiskMarketID (string | null)
- negRiskFeeBips (number | null)
- gmpChartMode (string | null)

Notes
- Numeric fields may appear as strings for some entries; normalize downstream.
- Dates are ISO datetimes; handle timezone suffix `Z`.

### Tag object (Event.tags)
- id (string)
- label (string)
- slug (string)
- publishedAt (string)
- createdAt (string)
- updatedAt (string)
- Optional flags: forceShow (boolean)

### Market object (Event.markets)
Represents a single tradable question (binary Yes/No) under the event.

Core identifiers
- id (string) — market id (group-local); not the on-chain condition hash
- conditionId (string) — CTF condition identifier (use this for prices/trades)
- slug (string)

Core metadata
- question (string)
- description (string)
- startDate, endDate (string, ISO datetime)
- startDateIso, endDateIso (string, `YYYY-MM-DD`)
- category (string) — not always present

State and flags
- active (boolean)
- closed (boolean)
- marketType (string)
- archived (boolean)
- new, featured (boolean)
- wideFormat (boolean)
- restricted (boolean)
- ready, funded, approved (boolean)

Economics and book
- liquidity, liquidityNum, liquidityAmm, liquidityClob (number | string)
- volume, volumeNum (number | string)
- volume24hr/1wk/1mo/1yr; volume24hrAmm/Clob etc. (number | string)
- makerBaseFee, takerBaseFee (number)
- orderPriceMinTickSize (number)
- orderMinSize (number)
- bestBid, bestAsk, lastTradePrice (number)
- spread (number)

CLOB/UMA linkage
- clobTokenIds (string-encoded JSON array of two ERC‑1155 token ids for outcomes)
- questionID (string)
- umaEndDate (string)
- umaResolutionStatus / umaResolutionStatuses (string or array-encoded string)
- umaBond, umaReward (string)

Administrative
- createdAt, updatedAt, closedTime (string)
- submitted_by (string)
- resolvedBy (string)
- notificationsEnabled, commentsEnabled (boolean)
- rfqEnabled, holdingRewardsEnabled (boolean)

Notes
- Many numeric fields are strings; coerce to numeric as needed.
- `conditionId` is the canonical key for historical timeseries and trades APIs:
  - Prices: `/prices-history?market=<conditionId>`
  - Trades: `/trades?market=<conditionId>` (some deployments also accept `conditionId=<...>`)

### Minimal join keys
- Event → Markets: use `Event.id` and the embedding (no explicit foreign key in market objects).
- Market → Trades/Prices: use `Market.conditionId`.

### Example (truncated)
```json
{
  "id": "903193",
  "slug": "presidential-election-winner-2024",
  "title": "Presidential Election Winner 2024",
  "volume": 3686335000.0,
  "markets": [
    {
      "id": "253591",
      "question": "Will Donald Trump win the 2024 US Presidential Election?",
      "conditionId": "0x...",
      "active": true,
      "closed": true,
      "endDate": "2024-11-05T12:00:00Z"
    }
  ]
}
```

### Parsing tips
- Treat ids (`Event.id`, `Market.id`, `Market.conditionId`) as strings consistently.
- When flattening, prefix columns to avoid collisions (e.g., `event_id`, `market_id`, `condition_id`).
- Parse `clobTokenIds` if outcome token ids are needed.



