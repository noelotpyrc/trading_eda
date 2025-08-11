#!/usr/bin/env python3
"""
Fetch price history for all markets from Polymarket events data.
Collects raw price data for both YES and NO outcomes.
"""

import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_history_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PolymarketPriceHistoryFetcher:
    def __init__(self, output_dir: str = "data/price_history", rate_limit: float = 0.5):
        """
        Initialize the price history fetcher.
        
        Args:
            output_dir: Directory to save price history data
            rate_limit: Seconds to wait between API calls
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PolymarketAnalysis/1.0',
            'Accept': 'application/json'
        })
        
        # Stats tracking
        self.stats = {
            'total_events': 0,
            'total_markets': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'empty_responses': 0,
            'api_errors': 0
        }
    
    def load_events_data(self, events_file: str) -> List[Dict]:
        """Load events data from JSON file."""
        logger.info(f"Loading events from {events_file}")
        
        with open(events_file, 'r') as f:
            events = json.load(f)
        
        logger.info(f"Loaded {len(events)} events")
        self.stats['total_events'] = len(events)
        return events
    
    def extract_markets_from_events(self, events: List[Dict]) -> List[Dict]:
        """Extract all markets with their metadata from events."""
        markets = []
        
        for event in events:
            event_id = event.get('id')
            event_title = event.get('title', 'Unknown Event')
            
            # Extract markets from the event
            event_markets = event.get('markets', [])
            
            for market in event_markets:
                # Parse clobTokenIds (can be string or array)
                clob_token_ids = market.get('clobTokenIds', [])
                if isinstance(clob_token_ids, str):
                    try:
                        clob_token_ids = json.loads(clob_token_ids)
                    except (json.JSONDecodeError, TypeError):
                        clob_token_ids = []
                
                market_info = {
                    'event_id': event_id,
                    'event_title': event_title,
                    'market_id': market.get('id'),
                    'condition_id': market.get('conditionId'),
                    'question': market.get('question', 'Unknown Question'),
                    'clob_token_ids': clob_token_ids,
                    'outcomes': market.get('outcomes', ['YES', 'NO']),
                    'closed': market.get('closed', False),
                    'resolved': market.get('resolved', False),
                    'end_date': market.get('endDate'),
                    'volume': market.get('volume', 0)
                }
                markets.append(market_info)
        
        logger.info(f"Extracted {len(markets)} markets from events")
        self.stats['total_markets'] = len(markets)
        return markets
    
    def resolve_clob_token_ids(self, condition_id: str, clob_token_ids: List[str]) -> List[str]:
        """
        Resolve CLOB token IDs for a condition.
        
        Args:
            condition_id: The condition ID
            clob_token_ids: Token IDs from event data
            
        Returns:
            List of verified token IDs
        """
        # If we have clobTokenIds from the event, use them
        if clob_token_ids and len(clob_token_ids) >= 2:
            return clob_token_ids[:2]  # YES, NO
        
        # Fallback: Query CLOB markets endpoint
        try:
            time.sleep(self.rate_limit)
            resp = self.session.get(
                "https://clob.polymarket.com/markets",
                params={"limit": 1000},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            markets = data.get('data', [])
            for market in markets:
                if market.get('condition_id') == condition_id:
                    tokens = market.get('tokens', [])
                    if len(tokens) >= 2:
                        return [tokens[0].get('token_id'), tokens[1].get('token_id')]
            
        except Exception as e:
            logger.warning(f"Failed to resolve token IDs for {condition_id}: {e}")
        
        return []
    
    def fetch_price_history(self, token_id: str, fidelity: int = 720) -> Optional[pd.DataFrame]:
        """
        Fetch price history for a single token.
        
        Args:
            token_id: CLOB token ID
            fidelity: Resolution in minutes (default 720 = 12 hours)
            
        Returns:
            DataFrame with price history or None if failed
        """
        try:
            time.sleep(self.rate_limit)
            resp = self.session.get(
                "https://clob.polymarket.com/prices-history",
                params={
                    "market": token_id,
                    "interval": "max",
                    "fidelity": fidelity
                },
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract history
            if isinstance(data, dict) and "history" in data:
                history = data["history"]
            elif isinstance(data, list):
                history = data
            else:
                logger.warning(f"Unexpected response format for token {token_id}")
                self.stats['empty_responses'] += 1
                return None
            
            if not history:
                logger.info(f"Empty price history for token {token_id}")
                self.stats['empty_responses'] += 1
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            
            # Standardize column names
            if 'timestamp' in df.columns and 'price' in df.columns:
                df.rename(columns={'timestamp': 't', 'price': 'p'}, inplace=True)
            
            if 't' in df.columns:
                df['datetime'] = pd.to_datetime(df['t'], unit='s', errors='coerce')
                df = df.dropna(subset=['datetime'])
                df = df.sort_values('datetime')
            
            self.stats['successful_fetches'] += 1
            return df
            
        except requests.RequestException as e:
            logger.error(f"API error fetching price history for token {token_id}: {e}")
            self.stats['api_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching price history for token {token_id}: {e}")
            self.stats['failed_fetches'] += 1
            return None
    

    
    def save_market_data(self, market_info: Dict, yes_df: pd.DataFrame, no_df: pd.DataFrame):
        """Save price history data for a market."""
        condition_id = market_info['condition_id']
        market_dir = self.output_dir / condition_id
        market_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'market_info': market_info,
            'fetched_at': datetime.now().isoformat(),
            'yes_data_points': len(yes_df) if not yes_df.empty else 0,
            'no_data_points': len(no_df) if not no_df.empty else 0
        }
        
        with open(market_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save raw price data
        if not yes_df.empty:
            yes_df.to_csv(market_dir / "yes_price_history.csv", index=False)
        
        if not no_df.empty:
            no_df.to_csv(market_dir / "no_price_history.csv", index=False)
    
    def fetch_all_price_histories(self, events_file: str, max_markets: Optional[int] = None):
        """
        Main method to fetch price histories for all markets.
        
        Args:
            events_file: Path to events JSON file
            max_markets: Limit number of markets to process (for testing)
        """
        logger.info("Starting price history fetch process")
        
        # Load events and extract markets
        events = self.load_events_data(events_file)
        markets = self.extract_markets_from_events(events)
        
        if max_markets:
            markets = markets[:max_markets]
            logger.info(f"Limited to {max_markets} markets for testing")
        
        # Process each market
        for i, market in enumerate(markets, 1):
            condition_id = market['condition_id']
            logger.info(f"Processing market {i}/{len(markets)}: {condition_id}")
            logger.info(f"Question: {market['question'][:100]}...")
            
            # Resolve token IDs
            token_ids = self.resolve_clob_token_ids(
                condition_id, 
                market['clob_token_ids']
            )
            
            if len(token_ids) < 2:
                logger.warning(f"Could not resolve token IDs for {condition_id}")
                self.stats['failed_fetches'] += 1
                continue
            
            yes_token_id, no_token_id = token_ids[0], token_ids[1]
            
            # Fetch price history for both outcomes
            logger.info(f"Fetching YES token: {yes_token_id}")
            yes_df = self.fetch_price_history(yes_token_id)
            
            logger.info(f"Fetching NO token: {no_token_id}")
            no_df = self.fetch_price_history(no_token_id)
            
            # Save data if we got anything
            yes_has_data = yes_df is not None and not yes_df.empty
            no_has_data = no_df is not None and not no_df.empty
            
            if yes_has_data or no_has_data:
                yes_data = yes_df if yes_has_data else pd.DataFrame()
                no_data = no_df if no_has_data else pd.DataFrame()
                self.save_market_data(market, yes_data, no_data)
                logger.info(f"Saved data for {condition_id}")
            else:
                logger.warning(f"No price data available for {condition_id}")
            
            # Progress update
            if i % 10 == 0:
                self.print_stats()
        
        logger.info("Price history fetch completed")
        self.print_stats()
        self.save_final_stats()
    
    def print_stats(self):
        """Print current statistics."""
        logger.info("=== FETCH STATISTICS ===")
        for key, value in self.stats.items():
            logger.info(f"{key}: {value}")
        
        if self.stats['total_markets'] > 0:
            success_rate = (self.stats['successful_fetches'] / (self.stats['total_markets'] * 2)) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
    
    def save_final_stats(self):
        """Save final statistics to file."""
        stats_file = self.output_dir / "fetch_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump({
                **self.stats,
                'completed_at': datetime.now().isoformat()
            }, f, indent=2)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Polymarket price histories")
    parser.add_argument(
        "--events-file", 
        required=True, 
        help="Path to events JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/price_history",
        help="Output directory for price data"
    )
    parser.add_argument(
        "--max-markets", 
        type=int, 
        help="Maximum number of markets to process (for testing)"
    )
    parser.add_argument(
        "--rate-limit", 
        type=float, 
        default=0.5,
        help="Seconds to wait between API calls"
    )
    
    args = parser.parse_args()
    
    # Validate events file exists
    if not os.path.exists(args.events_file):
        logger.error(f"Events file not found: {args.events_file}")
        return 1
    
    # Initialize fetcher and run
    fetcher = PolymarketPriceHistoryFetcher(
        output_dir=args.output_dir,
        rate_limit=args.rate_limit
    )
    
    try:
        fetcher.fetch_all_price_histories(
            events_file=args.events_file,
            max_markets=args.max_markets
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        fetcher.print_stats()
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
