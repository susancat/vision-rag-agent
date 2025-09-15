#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch long-term daily data for ETH + top N coins.

- CoinGecko (Demo API): last 365 days (price, market_cap, total_volume)
- Binance (no API key): backfill older daily OHLC -> use close as price, and estimate USD volume = close * base_volume
- Merge into one CSV per coin under data/docs/markets_combined/<coin_id>.csv
  Schema: date, price, market_cap, total_volume, source

Usage:
  python scripts/fetch_market_data.py --top 10 --vs usd --years 8

.env.local or .env should contain:
  COINGECKO_DEMO_API_KEY=your_demo_key
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

# ---------- dotenv (robust path) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_LOCAL = PROJECT_ROOT / ".env.local"
ENV = PROJECT_ROOT / ".env"
if ENV_LOCAL.exists():
    load_dotenv(ENV_LOCAL)
else:
    load_dotenv(ENV)

COINGECKO_ROOT = "https://api.coingecko.com/api/v3"
BINANCE_ROOT = "https://api.binance.com/api/v3/klines"

OUT_CG = PROJECT_ROOT / "data" / "docs" / "markets"           # 原始 CG（最近一年）
OUT_MERGED = PROJECT_ROOT / "data" / "docs" / "markets_combined"  # 合併後
OUT_CG.mkdir(parents=True, exist_ok=True)
OUT_MERGED.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("COINGECKO_DEMO_API_KEY")  # required for Demo API

# 排除常見穩定幣
EXCLUDE_IDS = {
    "tether", "usd-coin", "dai", "binance-usd", "true-usd", "first-digital-usd",
    "usdd", "frax", "paxos-standard", "pax-dollar", "gemini-dollar"
}
EXCLUDE_SYMBOLS = {"usdt", "usdc", "dai", "busd", "tusd", "fdusd", "usdd", "frax", "pax", "gusd"}

# CoinGecko id -> Binance symbol (USDT pairs)
BINANCE_SYMBOL_MAP: Dict[str, str] = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "cardano": "ADAUSDT",
    "dogecoin": "DOGEUSDT",
    "toncoin": "TONUSDT",
    "tron": "TRXUSDT",
    "avalanche-2": "AVAXUSDT",
}

# ---------- Helpers ----------
def to_unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def ymd(ts_ms: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(ts_ms/1000)))

def cg_get(path: str, params: Dict | None = None, retry: int = 3):
    if not API_KEY:
        raise RuntimeError("Missing COINGECKO_DEMO_API_KEY in .env.local or .env")
    url = f"{COINGECKO_ROOT}{path}"
    headers = {"x-cg-demo-api-key": API_KEY}
    params = dict(params or {})
    params.setdefault("x_cg_demo_api_key", API_KEY)  # fallback as query
    last_exc = None
    for attempt in range(1, retry + 1):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 429 and attempt < retry:
            sleep_s = 2 ** attempt
            print(f"[warn] CG 429 rate limited, wait {sleep_s}s")
            time.sleep(sleep_s); continue
        if r.status_code == 401:
            raise RuntimeError(f"CG 401: {r.text[:300]}")
        try:
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < retry:
                time.sleep(1.5 * attempt)
    raise last_exc

def list_top_coins(n: int, vs: str = "usd") -> List[Dict]:
    data = cg_get("/coins/markets", {
        "vs_currency": vs,
        "order": "market_cap_desc",
        "per_page": max(n + 5, 15),
        "page": 1,
        "sparkline": "false",
    })
    coins = []
    for c in data:
        cid = (c.get("id") or "").lower()
        sym = (c.get("symbol") or "").lower()
        if cid in EXCLUDE_IDS or sym in EXCLUDE_SYMBOLS:
            continue
        coins.append({
            "id": cid,
            "symbol": sym,
            "name": c.get("name"),
            "market_cap_rank": c.get("market_cap_rank"),
        })
        if len(coins) >= n:
            break
    return coins

def fetch_cg_last365(coin_id: str, vs: str) -> List[Dict]:
    """最近 365 天（日線）"""
    end = datetime.utcnow()
    start = end - timedelta(days=365)
    j = cg_get(f"/coins/{coin_id}/market_chart/range", {
        "vs_currency": vs,
        "from": to_unix(start),
        "to": to_unix(end),
    })
    prices = j.get("prices", [])
    mcaps = j.get("market_caps", [])
    vols = j.get("total_volumes", [])

    def to_map(arr):
        d = {}
        for ts, val in arr:
            d[ymd(ts)] = val
        return d

    pmap, mmap, vmap = to_map(prices), to_map(mcaps), to_map(vols)
    dates = sorted(set(pmap) | set(mmap) | set(vmap))
    rows = []
    for dt in dates:
        rows.append({
            "date": dt,
            "price": pmap.get(dt, ""),
            "market_cap": mmap.get(dt, ""),
            "total_volume": vmap.get(dt, ""),
            "source": "coingecko",
        })
    # 也輸出一份純 CG 的（可選）
    out = OUT_CG / f"{coin_id}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date","price","market_cap","total_volume"])
        w.writeheader(); w.writerows([{
            "date": r["date"], "price": r["price"],
            "market_cap": r["market_cap"], "total_volume": r["total_volume"]
        } for r in rows])
    return rows

def fetch_binance_all(symbol: str) -> List[Dict]:
    """抓該交易對所有可得的日K（分批 1000 根），回傳 list[dict(date, close, base_vol)]"""
    out = []
    limit = 1000
    # 從早期某個固定日期開始（2017-01-01）
    start_ms = int(datetime(2017,1,1, tzinfo=timezone.utc).timestamp() * 1000)
    while True:
        params = {"symbol": symbol, "interval": "1d", "limit": limit, "startTime": start_ms}
        r = requests.get(BINANCE_ROOT, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        for k in data:
            open_time = int(k[0])
            close = float(k[4])
            base_vol = float(k[5])  # base asset volume
            out.append({
                "date": ymd(open_time),
                "close": close,
                "base_vol": base_vol,
            })
        # 下一批開始時間：最後一根 K 的 close time + 1ms（k[6] 是 closeTime）
        next_start = int(data[-1][6]) + 1
        if next_start <= start_ms or len(data) < limit:
            break
        start_ms = next_start
        time.sleep(0.35)  # 禮貌點
    return out

def merge_series(coin_id: str, rows_cg: List[Dict], rows_bi: List[Dict], prefer_cg_from: str) -> List[Dict]:
    """
    合併：對於 prefer_cg_from 這天（含）之後用 CG；更早用 Binance。
    Binance 估算 USD 成交量：price_close * base_volume（USDT 交易對近似 USD）
    """
    merged: Dict[str, Dict] = {}
    # 先放 Binance 舊資料
    for r in rows_bi:
        est_price = r["close"]
        est_vol_usd = r["close"] * r["base_vol"] if not math.isnan(r["base_vol"]) else ""
        merged[r["date"]] = {
            "date": r["date"],
            "price": est_price,
            "market_cap": "",
            "total_volume": est_vol_usd,
            "source": "binance",
        }
    # 再放 CG 新資料（覆蓋同日）
    for r in rows_cg:
        if r["date"] >= prefer_cg_from:
            merged[r["date"]] = r
    rows = [merged[k] for k in sorted(merged.keys())]
    # 寫檔
    out = OUT_MERGED / f"{coin_id}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date","price","market_cap","total_volume","source"])
        w.writeheader(); w.writerows(rows)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="Top-N coins (ex-stables)")
    ap.add_argument("--vs", type=str, default="usd", help="Quote currency for CoinGecko (usd/eur/twd...)")
    ap.add_argument("--years", type=int, default=8, help="Backfill years via Binance (older than last 365d)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep between coins")
    args = ap.parse_args()

    coins = list_top_coins(args.top, args.vs)
    if not any(c["id"] == "ethereum" for c in coins):
        coins.insert(0, {"id": "ethereum","symbol":"eth","name":"Ethereum","market_cap_rank":2})

    meta_path = OUT_MERGED / "tokens.json"
    meta_path.write_text(json.dumps(coins, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[meta] wrote {meta_path}")

    prefer_cg_from = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    for c in coins:
        cid = c["id"]
        print(f"\n[coin] {cid}")
        # 1) 最近一年（CG）
        try:
            cg_rows = fetch_cg_last365(cid, args.vs)
            print(f"  CG: {len(cg_rows)} rows (last 365d)")
        except Exception as e:
            print(f"  !! CG failed for {cid}: {e}")
            cg_rows = []

        # 2) 回補（Binance）
        sym = BINANCE_SYMBOL_MAP.get(cid)
        bi_rows = []
        if sym:
            try:
                bi_rows = fetch_binance_all(sym)
                print(f"  Binance: {len(bi_rows)} rows (historical)")
            except Exception as e:
                print(f"  !! Binance failed for {cid} ({sym}): {e}")
        else:
            print(f"  (skip Binance backfill — no symbol mapping for {cid})")

        # 3) 合併
        merged = merge_series(cid, cg_rows, bi_rows, prefer_cg_from)
        print(f"  -> merged saved: {OUT_MERGED / (cid + '.csv')} ({len(merged)} rows)")
        time.sleep(args.sleep)

    print("\n[done] Combined series saved under data/docs/markets_combined/")
    print("      Ingest with: python api/ingest.py --docs data/docs --rebuild")

if __name__ == "__main__":
    main()
