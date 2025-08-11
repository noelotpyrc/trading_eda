# Polymarket API Research Documentation

## Overview

Polymarket operates a dual API architecture providing comprehensive access to prediction market data. This document outlines the complete API ecosystem, endpoints, data structures, limitations, and best practices discovered through research and testing.

## API Architecture

### Gamma API
- **Base URL**: `https://gamma-api.polymarket.com`
- **Purpose**: Market metadata, events, and market information
- **Primary Use**: Event discovery, market structure analysis
- **Authentication**: None required for public endpoints

### Data API  
- **Base URL**: `https://data-api.polymarket.com`
- **Purpose**: Trading data, price history, and market activity
- **Primary Use**: Historical analysis, trade records, price movements
- **Authentication**: None required for public endpoints

## Core Endpoints

### Events API (`/events`)

Retrieves prediction market events with comprehensive filtering capabilities.

**Endpoint**: `GET https://gamma-api.polymarket.com/events`

**Key Parameters**:
```
limit: integer                 # Number of results (default: 100)
offset: integer               # Pagination offset
order: string                 # Sort key (volume, liquidity, startDate, etc.)
ascending: boolean            # Sort direction (default: false)
id: string                    # Specific event ID
slug: string                  # Event slug identifier
archived: boolean             # Include archived events
active: boolean               # Filter by active status
closed: boolean               # Filter by resolved/closed events
liquidity_min: number         # Minimum liquidity threshold
liquidity_max: number         # Maximum liquidity threshold
volume_min: number            # Minimum volume threshold  
volume_max: number            # Maximum volume threshold
start_date_min: string        # Start date filter (YYYY-MM-DD)
start_date_max: string        # Start date filter (YYYY-MM-DD)
end_date_min: string          # End date filter (YYYY-MM-DD)
end_date_max: string          # End date filter (YYYY-MM-DD)
tag: string                   # Filter by tag label
tag_id: string                # Filter by tag ID
related_tags: boolean         # Include related tags
tag_slug: string              # Filter by tag slug
```

**Response Structure** (58+ fields per event):
```json
{
  "id": "string",
  "ticker": "string", 
  "slug": "string",
  "title": "string",
  "description": "string",
  "resolutionSource": "string",
  "startDate": "ISO datetime",
  "creationDate": "ISO datetime", 
  "endDate": "ISO datetime",
  "image": "URL string",
  "icon": "URL string",
  "active": boolean,
  "closed": boolean,
  "archived": boolean,
  "new": boolean,
  "featured": boolean,
  "restricted": boolean,
  "liquidity": "number|string",
  "volume": "number|string", 
  "openInterest": "number|string",
  "volume24hr": number,
  "volume1wk": number,
  "volume1mo": number,
  "volume1yr": number,
  "liquidityAmm": number,
  "liquidityClob": number,
  "markets": [Market],
  "tags": [Tag],
  "competitive": number,
  "commentsEnabled": boolean,
  "commentCount": number,
  "enableOrderBook": boolean,
  "cyom": boolean,
  "closedTime": "string",
  "showAllOutcomes": boolean,
  "showMarketImages": boolean,
  "enableNegRisk": boolean,
  "negRiskAugmented": boolean,
  "pendingDeployment": boolean,
  "deploying": boolean,
  "updatedBy": "number|string|null",
  "automaticallyResolved": "boolean|null",
  "automaticallyActive": "boolean|null",
  "featuredImage": "string|null",
  "negRisk": "boolean|null",
  "negRiskMarketID": "string|null",
  "negRiskFeeBips": "number|null",
  "gmpChartMode": "string|null"
}
```

### Markets API (`/markets`)

Retrieves individual market information.

**Endpoint**: `GET https://gamma-api.polymarket.com/markets`

**Key Parameters**:
```
event_id: integer             # Filter by parent event
closed: boolean               # Filter by market status
limit: integer                # Results limit
offset: integer               # Pagination offset
```

**Market Object Structure**:
```json
{
  "id": "string",
  "conditionId": "string",
  "slug": "string", 
  "question": "string",
  "description": "string",
  "startDate": "ISO datetime",
  "endDate": "ISO datetime",
  "startDateIso": "YYYY-MM-DD",
  "endDateIso": "YYYY-MM-DD",
  "category": "string",
  "active": boolean,
  "closed": boolean,
  "marketType": "string",
  "archived": boolean,
  "new": boolean,
  "featured": boolean,
  "wideFormat": boolean,
  "restricted": boolean,
  "ready": boolean,
  "funded": boolean,
  "approved": boolean,
  "liquidity": "number|string",
  "liquidityNum": "number|string",
  "liquidityAmm": "number|string", 
  "liquidityClob": "number|string",
  "volume": "number|string",
  "volumeNum": "number|string",
  "volume24hr": "number|string",
  "volume1wk": "number|string",
  "volume1mo": "number|string",
  "volume1yr": "number|string",
  "volume1wkAmm": "number|string",
  "volume1moAmm": "number|string",
  "volume1yrAmm": "number|string",
  "volume1wkClob": "number|string",
  "volume1moClob": "number|string",
  "volume1yrClob": "number|string",
  "makerBaseFee": number,
  "takerBaseFee": number,
  "orderPriceMinTickSize": number,
  "orderMinSize": number,
  "bestBid": number,
  "bestAsk": number,
  "lastTradePrice": number,
  "spread": number,
  "clobTokenIds": "string",
  "questionID": "string",
  "umaEndDate": "string",
  "umaResolutionStatus": "string",
  "umaResolutionStatuses": "string|array",
  "umaBond": "string",
  "umaReward": "string",
  "createdAt": "string",
  "updatedAt": "string",
  "closedTime": "string",
  "submitted_by": "string",
  "resolvedBy": "string",
  "notificationsEnabled": boolean,
  "commentsEnabled": boolean,
  "rfqEnabled": boolean,
  "holdingRewardsEnabled": boolean
}
```

### Price History API (`/prices-history`)

Provides historical price timeseries data for traded tokens.

**Endpoint**: `GET https://data-api.polymarket.com/prices-history`

**Parameters**:
```
market: string                # CLOB token ID (conditionId)
startTs: integer              # Start timestamp (Unix UTC)
endTs: integer                # End timestamp (Unix UTC)
interval: string              # Predefined intervals: 1m,1w,1d,6h,1h,max
fidelity: integer             # Resolution in minutes
```

**Response Structure**:
```json
[
  {
    "timestamp": "ISO datetime",
    "price": number,
    "volume": number
  }
]
```

### Trades API (`/trades`)

Accesses individual trade records across markets and users.

**Endpoint**: `GET https://data-api.polymarket.com/trades`

**Parameters**:
```
user: string                  # User wallet address
limit: integer                # Max trades per request (default: 100, max: 500)
offset: integer               # Pagination offset
takerOnly: boolean            # Return only taker orders (default: true)
filterType: string            # Filter by type: CASH or TOKENS
filterAmount: number          # Amount filter related to filterType
market: string                # Condition ID (supports comma-separated)
conditionId: string           # Alternative parameter name for market
side: string                  # Trade side: BUY or SELL
```

**Trade Object Structure**:
```json
{
  "id": "string",
  "timestamp": "ISO datetime",
  "market": "string",
  "conditionId": "string", 
  "side": "string",
  "size": "string",
  "price": "string",
  "feeRateBps": number,
  "fee": "string",
  "status": "string",
  "type": "string",
  "user": "string",
  "counterparty": "string"
}
```

## Data Structures

### Tag Object
```json
{
  "id": "string",
  "label": "string",
  "slug": "string", 
  "publishedAt": "string",
  "createdAt": "string",
  "updatedAt": "string",
  "forceShow": boolean
}
```

### Key Identifiers

- **Event ID**: String identifier for events (e.g., "903173")
- **Market ID**: String identifier for individual markets within events
- **Condition ID**: Canonical identifier for CLOB markets, used for price/trade queries
- **Slug**: Human-readable URL identifier

### Data Type Inconsistencies

**Important**: Many numeric fields are returned as strings and require type coercion:
- `volume`, `liquidity`, `openInterest` may be strings
- `clobTokenIds` is JSON-encoded string array
- Timestamps mix ISO datetime and Unix formats

## API Limitations & Constraints

### Rate Limiting
- **No official rate limits documented**
- **Trades API**: Hard limit of 500 records per request
- **Recommended**: Implement exponential backoff and request throttling
- **Best Practice**: Add delays between requests (1-2 seconds)

### Pagination
- **Standard**: offset/limit pagination across all endpoints  
- **Events API**: Supports large result sets (1000+ per request)
- **Trades API**: Limited to 500 records maximum per request
- **Data Completeness**: Large date ranges require chunking strategies

### Parameter Variations
- **Trades API**: Accepts both `market` and `conditionId` parameters
- **Date Formats**: Mix of ISO strings and Unix timestamps
- **Boolean Values**: Some endpoints accept string representations

### Data Quality Issues
- **Duplicate Records**: Overlapping time windows return duplicate events
- **Missing Fields**: Not all fields present in every response
- **Type Inconsistencies**: Numeric data sometimes returned as strings

## Best Practices

### Error Handling
```python
def robust_api_call(url, params, max_retries=3, timeout=30):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Pagination Strategy
```python
def paginate_endpoint(base_params, max_pages=10):
    results = []
    offset = 0
    page_size = 500  # Adjust based on endpoint
    
    for page in range(max_pages):
        params = {**base_params, "limit": page_size, "offset": offset}
        batch = api_call(params)
        
        if not batch:
            break
            
        results.extend(batch)
        
        if len(batch) < page_size:
            break
            
        offset += page_size
    
    return results
```

### Date Range Chunking
```python
def chunk_date_range(start_date, end_date, chunk_days=14):
    chunks = []
    current = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))
        current = chunk_end + timedelta(days=1)
    
    return chunks
```

### Data Validation
```python
def validate_event_data(events):
    required_fields = ['id', 'slug', 'volume', 'startDate', 'endDate']
    valid_events = []
    
    for event in events:
        if all(field in event for field in required_fields):
            # Type coercion
            try:
                event['volume'] = float(event['volume'])
                valid_events.append(event)
            except (ValueError, TypeError):
                continue
                
    return valid_events
```

## Alternative Data Sources

### On-Chain Data
- **Dune Analytics**: Complete trade history without API rate limits
- **Flipside Crypto**: Additional blockchain analytics capabilities
- **Advantage**: No pagination limits, complete historical coverage

### Third-Party Services
- **PolyData Explorer**: Pre-cleaned datasets with JSON export
- **polymarket-py**: Python package for simplified data access
- **Polymarket Data Service**: NestJS-based aggregation service

### On-Chain Contract Details
- **CLOB (Central Limit Order Book)**: ERC-1155 based conditional tokens
- **UMA Protocol**: Resolution mechanism for markets
- **CTF (Conditional Token Framework)**: Underlying token standard

## Common Use Cases

### Event Discovery
```python
# Find high-volume resolved events in date range
params = {
    "closed": True,
    "volume_min": 10000,
    "start_date_min": "2024-01-01",
    "start_date_max": "2024-12-31",
    "order": "volume",
    "ascending": False,
    "limit": 1000
}
```

### Market Analysis
```python
# Get detailed market structure for specific event
event_markets = get_json("/events", {"id": event_id})
for market in event_markets[0]["markets"]:
    condition_id = market["conditionId"]
    # Fetch price history and trades using condition_id
```

### Historical Price Analysis
```python
# High-resolution price data for backtesting
params = {
    "market": condition_id,
    "startTs": start_timestamp,
    "endTs": end_timestamp,
    "fidelity": 60  # 1-minute resolution
}
```

### Trade Flow Analysis
```python
# Comprehensive trade data for market microstructure
params = {
    "market": condition_id,
    "limit": 500,
    "takerOnly": False,  # Include maker orders
    "offset": 0
}
```

## Technical Notes

### CLOB Token IDs
- Each binary market has two ERC-1155 tokens (YES/NO outcomes)
- `clobTokenIds` contains JSON-encoded array of token IDs
- Used for on-chain analysis and order book operations

### Resolution Mechanisms
- **UMA Protocol**: Decentralized oracle for dispute resolution
- **Manual Resolution**: Polymarket team resolution for some markets
- **Automatic Resolution**: API-based resolution for objective outcomes

### Market Types
- **Binary**: Yes/No outcomes (most common)
- **Categorical**: Multiple exclusive outcomes
- **Scalar**: Range-based outcomes (less common)

## Research Sources

- Official Polymarket API Documentation
- Community-developed tools and packages
- On-chain contract analysis
- Third-party data aggregation services
- Practical testing and implementation validation


