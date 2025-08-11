#!/usr/bin/env python3
"""
Resumable Polymarket trades backfiller using the official Data API.

Features:
- Reads conditionIds from an events JSON file (Gamma export)
- Paginates trades with limit/offset (500/page)
- Global rate limiting and respectful retries (429, 5xx)
- Writes JSONL per conditionId and maintains a checkpoint file
- Can resume from last checkpoint or infer from existing JSONL size

Usage (example):
  ./.venv/bin/python polymarket/fetch_trades_backfill.py \
    --events-file data/polymarket/events/gamma_events_test_jan2024.json \
    --output-dir polymarket/data/trades \
    --checkpoint-file polymarket/data/trades/checkpoint.json \
    --limit 500 --rps 2.0 --concurrency 2 --max-markets 5
"""

from __future__ import annotations

import json
import os
import time
import math
import signal
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import requests


DATA_API_BASE = "https://data-api.polymarket.com"
TRADES_ENDPOINT = "/trades"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("polymarket_trades_backfill")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = setup_logger()


class StopRequested(Exception):
    """Raised to cooperatively stop work on Ctrl+C."""
    pass


def str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    return None


class RateLimiter:
    """Simple global rate limiter allowing up to rps requests per second."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 0.1)
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self, stop_event: Optional[threading.Event] = None):
        with self._lock:
            now = time.monotonic()
            delay = self._last + self.min_interval - now
            if delay > 0:
                if stop_event is not None and stop_event.wait(delay):
                    raise StopRequested()
                now = time.monotonic()
            self._last = now


class CheckpointStore:
    """JSON checkpoint mapping conditionId -> {offset, complete, last_ts} """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                logger.warning("Checkpoint file is unreadable, starting fresh")

    def get(self, cid: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data.get(cid, {}))

    def update(self, cid: str, offset: int, complete: bool, last_ts: Optional[int] = None):
        with self._lock:
            state = self._data.get(cid, {})
            state["offset"] = offset
            state["complete"] = complete
            if last_ts is not None:
                state["last_ts"] = last_ts
            self._data[cid] = state
            self._flush()

    def _flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)


class TradesFetcher:
    def __init__(
        self,
        output_dir: Path,
        checkpoint: CheckpointStore,
        limit: int = 500,
        rps: float = 2.0,
        timeout: int = 30,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        taker_only: Optional[bool] = False,
        progress_every_pages: int = 50,
        max_rows_per_market: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = checkpoint
        self.limit = max(1, min(limit, 500))
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limiter = RateLimiter(rps)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PolymarketTradesBackfill/1.0"})
        self.stop_event = threading.Event()
        self.taker_only = taker_only
        self.progress_every_pages = max(1, int(progress_every_pages))
        self.max_rows_per_market = max_rows_per_market if (max_rows_per_market is None or max_rows_per_market > 0) else None

    def _http_get(self, path: str, params: Dict[str, Any]) -> requests.Response:
        # Global rate limit (interruptible)
        if self.stop_event.is_set():
            raise StopRequested()
        self.rate_limiter.acquire(self.stop_event)

        url = f"{DATA_API_BASE}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if self.stop_event.is_set():
                    raise StopRequested()
                resp = self.session.get(url, params=params, timeout=self.timeout)
                # Log the fully resolved request URL for traceability
                try:
                    logger.info(f"HTTP GET {resp.url} [attempt {attempt+1}] status={resp.status_code}")
                except Exception:
                    logger.info(f"HTTP GET {url} params={params} [attempt {attempt+1}] status={resp.status_code}")
                if resp.status_code == 429:
                    # Respect Retry-After if present
                    retry_after = resp.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after else self.backoff_base * (2 ** attempt)
                    logger.info(f"429 received, sleeping {wait_s:.2f}s")
                    if self.stop_event.wait(wait_s):
                        raise StopRequested()
                    continue
                if 500 <= resp.status_code < 600:
                    wait_s = self.backoff_base * (2 ** attempt)
                    logger.info(f"5xx received ({resp.status_code}), sleeping {wait_s:.2f}s")
                    if self.stop_event.wait(wait_s):
                        raise StopRequested()
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_error = e
                wait_s = self.backoff_base * (2 ** attempt)
                try:
                    logger.info(f"Request error on {url} with params={params}, retrying in {wait_s:.2f}s: {e}")
                except Exception:
                    logger.info(f"Request error, retrying in {wait_s:.2f}s: {e}")
                if self.stop_event.wait(wait_s):
                    raise StopRequested()

        assert last_error is not None
        raise last_error

    def _infer_existing_offset(self, file_path: Path) -> int:
        if not file_path.exists():
            return 0
        # Efficient-ish line count
        cnt = 0
        with file_path.open("rb") as f:
            for _ in f:
                cnt += 1
        return cnt

    def backfill_condition(self, condition_id: str) -> Tuple[str, int]:
        """Deprecated: kept for backward compatibility. Uses Data API incorrectly for some markets."""
        logger.warning("backfill_condition is deprecated. Prefer backfill_market with market_id.")
        out_file = self.output_dir / f"{condition_id}.jsonl"

        state = self.checkpoint.get(condition_id)
        offset = int(state.get("offset", 0))
        complete = bool(state.get("complete", False))

        if complete:
            logger.info(f"{condition_id}: already complete, skipping")
            return condition_id, 0

        # If resuming without checkpoint, infer from file line count
        if offset == 0 and out_file.exists():
            offset = self._infer_existing_offset(out_file)
            if offset:
                logger.info(f"{condition_id}: inferred offset from file = {offset}")

        appended_total = 0
        logger.info(f"{condition_id}: starting backfill at offset {offset}, page size {self.limit}")
        with out_file.open("a", encoding="utf-8") as fh:
            while True:
                if self.stop_event.is_set():
                    raise StopRequested()
                params = {
                    "conditionId": condition_id,
                    "limit": self.limit,
                    "offset": offset,
                }
                t0 = time.time()
                resp = self._http_get(TRADES_ENDPOINT, params)
                try:
                    batch = resp.json()
                except Exception as e:
                    logger.warning(f"{condition_id}: invalid JSON, aborting: {e}")
                    break

                if not isinstance(batch, list):
                    logger.warning(f"{condition_id}: unexpected response type {type(batch)}")
                    break

                if not batch:
                    # done
                    self.checkpoint.update(condition_id, offset, True)
                    logger.info(f"{condition_id}: complete at offset {offset}")
                    break

                # Append JSONL
                for row in batch:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                fh.flush()

                appended_total += len(batch)
                offset += len(batch)
                last_ts = _safe_max_ts(batch)
                elapsed_ms = int((time.time() - t0) * 1000)
                self.checkpoint.update(condition_id, offset, False, last_ts=last_ts)
                logger.info(
                    f"{condition_id}: fetched {len(batch)} trades in {elapsed_ms}ms, "
                    f"new offset {offset}, total appended {appended_total}, last_ts {last_ts}"
                )

                # If we got fewer than limit, we are done
                if len(batch) < self.limit:
                    self.checkpoint.update(condition_id, offset, True)
                    logger.info(f"{condition_id}: complete at offset {offset}")
                    break

        return condition_id, appended_total

    def backfill_market(
        self,
        market_id: str,
        condition_id: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Backfill one market using Data API 'market' param.

        Per Polymarket Data API behavior, the 'market' query param can be either
        the numeric/slug market id OR the conditionId hex. When a condition_id is
        provided, we pass 'market'=<condition_id> to ensure correct scoping.

        Returns (market_id, total_appended).
        """
        if not condition_id:
            logger.info(f"market {market_id}: missing condition_id; skipping")
            return market_id, 0

        key = f"mode:market|m:{market_id}|cid:{condition_id}"
        fname = f"market_{market_id}.jsonl" if not condition_id else f"market_{market_id}_{condition_id}.jsonl"
        out_file = self.output_dir / fname

        state = self.checkpoint.get(key)
        offset = int(state.get("offset", 0))
        complete = bool(state.get("complete", False))

        if complete:
            logger.info(f"market {market_id}: already complete, skipping")
            return market_id, 0

        if offset == 0 and out_file.exists():
            offset = self._infer_existing_offset(out_file)
            if offset:
                logger.info(f"market {market_id}: inferred offset from file = {offset}")

        appended_total = 0
        start_time = time.time()
        pages = 0
        logger.info(
            f"market {market_id}: starting backfill at page {offset}, page size {self.limit}, using market param={condition_id}"
        )
        last_page_sig: Optional[str] = None
        with out_file.open("a", encoding="utf-8") as fh:
            while True:
                if self.stop_event.is_set():
                    raise StopRequested()
                # Always use conditionId as market param
                params = {
                    "market": condition_id,
                    "limit": self.limit,
                    "offset": offset,
                    # Attempt to stabilize paging if supported
                    "ascending": "true",
                }
                if self.taker_only is not None:
                    params["takerOnly"] = "true" if self.taker_only else "false"
                t0 = time.time()
                resp = self._http_get(TRADES_ENDPOINT, params)
                try:
                    batch = resp.json()
                except Exception as e:
                    logger.warning(f"market {market_id}: invalid JSON, aborting: {e}")
                    break

                if not isinstance(batch, list):
                    logger.warning(f"market {market_id}: unexpected response type {type(batch)}")
                    break

                if not batch:
                    self.checkpoint.update(key, offset, True)
                    logger.info(f"market {market_id}: complete at page {offset}")
                    break

                # Validate/filter by condition if provided
                filtered = batch
                if condition_id:
                    expected = str(condition_id).lower()
                    filtered = [r for r in batch if str(r.get("conditionId", "")).lower() == expected]

                # Detect repeated page (server returning same data again) and stop (before writing)
                page_sig = hashlib.sha1(
                    ("|".join(
                        json.dumps(r, sort_keys=True, separators=(",", ":")) for r in filtered
                    )).encode("utf-8")
                ).hexdigest() if filtered else f"empty:{len(batch)}"
                if last_page_sig is not None and page_sig == last_page_sig:
                    total_elapsed = time.time() - start_time
                    logger.warning(
                        f"market {market_id}: current page identical to previous (signature {page_sig}). Marking complete to avoid loop."
                    )
                    self.checkpoint.update(key, offset, True, last_ts=_safe_max_ts(filtered))
                    break
                last_page_sig = page_sig

                # Write rows as-is (no cross-page dedup)
                for row in filtered:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                fh.flush()

                appended_total += len(filtered)
                # Advance to next page (offset is page index)
                offset += 1
                last_ts = _safe_max_ts(filtered)
                elapsed_ms = int((time.time() - t0) * 1000)
                self.checkpoint.update(key, offset, False, last_ts=last_ts)
                logger.info(
                    f"market {market_id}: fetched {len(batch)} trades in {elapsed_ms}ms, "
                    f"matched {len(filtered)} for condition {condition_id or 'N/A'}, "
                    f"new page {offset}, total appended {appended_total}, last_ts {last_ts}"
                )
                # Stop early if target reached
                if self.max_rows_per_market is not None and appended_total >= self.max_rows_per_market:
                    total_elapsed = time.time() - start_time
                    rows_per_sec = appended_total / total_elapsed if total_elapsed > 0 else 0.0
                    self.checkpoint.update(key, offset, True, last_ts=last_ts)
                    logger.info(
                        f"market {market_id}: target rows reached ({appended_total} >= {self.max_rows_per_market}). "
                        f"elapsed={total_elapsed:.1f}s, avg_rows_per_sec={rows_per_sec:.1f}"
                    )
                    break
                pages += 1
                if pages % self.progress_every_pages == 0:
                    total_elapsed = time.time() - start_time
                    rows_per_sec = appended_total / total_elapsed if total_elapsed > 0 else 0.0
                    req_per_sec = pages / total_elapsed if total_elapsed > 0 else 0.0
                    logger.info(
                        f"market {market_id}: progress pages={pages}, appended={appended_total}, "
                        f"elapsed={total_elapsed:.1f}s, avg_rows_per_sec={rows_per_sec:.1f}, avg_req_per_sec={req_per_sec:.2f}"
                    )

                if len(batch) < self.limit:
                    self.checkpoint.update(key, offset, True)
                    total_elapsed = time.time() - start_time
                    rows_per_sec = appended_total / total_elapsed if total_elapsed > 0 else 0.0
                    logger.info(
                        f"market {market_id}: complete at page {offset}. "
                        f"pages={pages}, appended={appended_total}, elapsed={total_elapsed:.1f}s, avg_rows_per_sec={rows_per_sec:.1f}"
                    )
                    break

        return market_id, appended_total


def _safe_max_ts(batch: List[Dict[str, Any]]) -> Optional[int]:
    max_ts = None
    for row in batch:
        ts = row.get("timestamp")
        try:
            val = int(ts) if ts is not None else None
        except Exception:
            val = None
        if val is not None:
            max_ts = val if max_ts is None else max(max_ts, val)
    return max_ts


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def load_markets_from_events(events_file: Path) -> List[Dict[str, str]]:
    data = json.loads(events_file.read_text())
    markets_list: List[Dict[str, str]] = []
    for evt in data:
        markets = evt.get("markets", [])
        for m in markets:
            mid = m.get("id")
            cid = m.get("conditionId")
            vol = _parse_float(m.get("volume"))
            if isinstance(mid, (str, int)):
                markets_list.append({
                    "market_id": str(mid),
                    "condition_id": str(cid) if isinstance(cid, str) else "",
                    "question": m.get("question", ""),
                    "volume": vol,
                })
    # Deduplicate by market_id, preserve order
    seen = set()
    ordered: List[Dict[str, str]] = []
    for item in markets_list:
        mid = item["market_id"]
        if mid not in seen:
            seen.add(mid)
            ordered.append(item)
    return ordered


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket trades backfiller (official Data API)")
    parser.add_argument("--events-file", required=True, help="Path to events JSON file")
    parser.add_argument("--output-dir", default="polymarket/data/trades", help="Directory for JSONL outputs")
    parser.add_argument("--checkpoint-file", default="polymarket/data/trades/checkpoint.json", help="Path to checkpoint JSON")
    parser.add_argument("--limit", type=int, default=500, help="Page size (max 500)")
    parser.add_argument("--rps", type=float, default=2.0, help="Global requests per second")
    parser.add_argument("--max-markets", type=int, help="Limit number of markets (for testing)")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this index in conditionId list")
    parser.add_argument("--only-market-id", action="append", help="Backfill only this market id (repeatable)")
    parser.add_argument("--only-condition-id", action="append", help="Filter markets by this condition id (repeatable)")
    parser.add_argument("--taker-only", type=str, default="false", help="If true, include only taker trades; default false includes maker+\n")
    parser.add_argument("--max-rows-per-market", type=int, help="Stop after fetching this many rows per market (early stop)")
    parser.add_argument("--progress-every-pages", type=int, default=50, help="Log a progress summary every N pages")
    parser.add_argument("--min-market-volume", type=float, help="Only process markets with volume >= this value (from events file)")

    args = parser.parse_args()

    events_file = Path(args.events_file)
    output_dir = Path(args.output_dir)
    checkpoint_file = Path(args.checkpoint_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    markets = load_markets_from_events(events_file)
    if args.start_index:
        markets = markets[args.start_index :]
    if args.max_markets:
        markets = markets[: args.max_markets]

    # Optional filtering by explicit ids
    if args.only_market_id:
        allow = set(str(x) for x in args.only_market_id)
        markets = [m for m in markets if m["market_id"] in allow]
    if args.only_condition_id:
        allowc = set(str(x) for x in args.only_condition_id)
        markets = [m for m in markets if m.get("condition_id") in allowc]

    # Filter by minimum volume if requested
    if args.min_market_volume is not None:
        before = len(markets)
        markets = [m for m in markets if float(m.get("volume", 0.0)) >= float(args.min_market_volume)]
        logger.info(f"Volume filter: min={args.min_market_volume} kept {len(markets)}/{before} markets")

    logger.info(f"Loaded {len(markets)} markets to backfill")

    checkpoint = CheckpointStore(checkpoint_file)
    fetcher = TradesFetcher(
        output_dir=output_dir,
        checkpoint=checkpoint,
        limit=args.limit,
        rps=args.rps,
        taker_only=str_to_bool(args.taker_only),
        progress_every_pages=args.progress_every_pages,
        max_rows_per_market=args.max_rows_per_market,
    )

    # Graceful shutdown
    def handle_sigint(sig, frame):
        logger.info("Received interrupt, requesting stop...")
        fetcher.stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    total_appended = 0
    # Sequential processing (no concurrency)
    for item in markets:
        if fetcher.stop_event.is_set():
            break
        mid = item["market_id"]
        cid = item.get("condition_id") or None
        try:
            _, appended = fetcher.backfill_market(mid, cid)
            total_appended += appended
            logger.info(f"market {mid}: appended {appended} trades")
        except StopRequested:
            logger.info(f"market {mid}: stopped by request")
            break
        except Exception as e:
            logger.error(f"market {mid}: failed with error: {e}")

    logger.info(f"Done. Total appended trades: {total_appended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


