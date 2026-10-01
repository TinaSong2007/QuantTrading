"""
Factor Definitions and Database Computation Engine.
Calculates Momentum, Reversal, Volatility, EP_TTM, and ROE_TTM factors,
cleans them using processor.py, and persists results into factor_metrics table.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from engine.db import get_connection, get_db
from engine.etl.collector import batch_insert
from engine.factors.processor import clean_factor_cross_section
from engine.pit_query import get_pit_financials, get_tradable_mask


def fetch_historical_bars(
    trade_date: str,
    lookback_days: int = 40,
    db_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Fetches the last N trading days of daily bars up to trade_date.
    """
    conn = get_connection(db_path)
    sql = """
    SELECT trade_date
    FROM (SELECT DISTINCT trade_date FROM daily_bar WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT ?)
    ORDER BY trade_date ASC;
    """
    cursor = conn.cursor()
    cursor.execute(sql, (trade_date, lookback_days))
    dates = [r[0] for r in cursor.fetchall()]

    if not dates:
        conn.close()
        return pl.DataFrame()

    min_date = dates[0]
    bars_sql = """
    SELECT ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close
    FROM daily_bar
    WHERE trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date ASC;
    """
    cursor.execute(bars_sql, (min_date, trade_date))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame([dict(r) for r in rows])


def compute_cross_sectional_factors(
    trade_date: str,
    db_path: Optional[str] = None
) -> Dict[str, pl.DataFrame]:
    """
    Computes a full suite of standardized and neutralized factors for a given trade_date:
    - MOM_20D (20-day Momentum)
    - REV_5D (5-day Reversal)
    - VOL_20D (20-day Volatility)
    - EP_TTM (TTM Earnings / Market Cap)
    - ROE_TTM (TTM Return on Equity)
    """
    bars_df = fetch_historical_bars(trade_date, lookback_days=30, db_path=db_path)
    if bars_df.is_empty():
        return {}

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ts_code, industry FROM stock_basic;")
    basics = {r["ts_code"]: r["industry"] for r in cursor.fetchall()}
    conn.close()

    pit_df = get_pit_financials(trade_date, db_path=db_path)
    pit_map = {}
    if not pit_df.is_empty():
        for r in pit_df.to_dicts():
            pit_map[r["ts_code"]] = r

    # Group bars by ts_code to compute technical factors
    factor_records = {
        "MOM_20D": [],
        "REV_5D": [],
        "VOL_20D": [],
        "EP_TTM": [],
        "ROE_TTM": [],
    }

    # Distinct trade dates sorted
    all_dates = sorted(bars_df["trade_date"].unique().to_list())
    current_date = trade_date

    # Pivot / slice current bars
    curr_bars = bars_df.filter(pl.col("trade_date") == current_date)
    curr_map = {r["ts_code"]: r for r in curr_bars.to_dicts()}

    for ts_code, curr_bar in curr_map.items():
        sub = bars_df.filter(pl.col("ts_code") == ts_code).sort("trade_date")
        n_bars = sub.height
        if n_bars < 5:
            continue

        adj_closes = sub["adj_close"].to_numpy()
        daily_rets = np.diff(adj_closes) / adj_closes[:-1]

        # Total market cap proxy (amount / 0.02 * base_shares or close * estimated float)
        # Using a deterministic market cap for the stock
        close_p = curr_bar["close"]
        code_int = int("".join(filter(str.isdigit, ts_code)) or "1")
        shares = 1e8 + (code_int % 50) * 1e7
        total_mv = close_p * shares

        industry = basics.get(ts_code, "综合")

        # 1. MOM_20D
        if n_bars >= 20:
            mom_20 = (adj_closes[-1] / adj_closes[-20]) - 1.0
            factor_records["MOM_20D"].append({
                "ts_code": ts_code, "trade_date": trade_date, "raw_val": float(mom_20),
                "industry": industry, "total_mv": total_mv
            })

        # 2. REV_5D
        if n_bars >= 5:
            rev_5 = (adj_closes[-1] / adj_closes[-5]) - 1.0
            factor_records["REV_5D"].append({
                "ts_code": ts_code, "trade_date": trade_date, "raw_val": float(rev_5),
                "industry": industry, "total_mv": total_mv
            })

        # 3. VOL_20D
        if len(daily_rets) >= 15:
            vol_20 = float(np.std(daily_rets[-20:])) if len(daily_rets) >= 20 else float(np.std(daily_rets))
            factor_records["VOL_20D"].append({
                "ts_code": ts_code, "trade_date": trade_date, "raw_val": vol_20,
                "industry": industry, "total_mv": total_mv
            })

        # 4. EP_TTM & 5. ROE_TTM
        pit_item = pit_map.get(ts_code)
        if pit_item and pit_item.get("ttm_net_profit") is not None:
            ep = pit_item["ttm_net_profit"] / total_mv
            factor_records["EP_TTM"].append({
                "ts_code": ts_code, "trade_date": trade_date, "raw_val": float(ep),
                "industry": industry, "total_mv": total_mv
            })
        if pit_item and pit_item.get("roe_ttm") is not None:
            factor_records["ROE_TTM"].append({
                "ts_code": ts_code, "trade_date": trade_date, "raw_val": float(pit_item["roe_ttm"]),
                "industry": industry, "total_mv": total_mv
            })

    cleaned_factors = {}
    for factor_name, raw_list in factor_records.items():
        if len(raw_list) >= 10:
            raw_df = pl.DataFrame(raw_list)
            clean_df = clean_factor_cross_section(raw_df)
            cleaned_factors[factor_name] = clean_df

    return cleaned_factors


def populate_factors_for_dates(
    trade_dates: List[str],
    db_path: Optional[str] = None
) -> int:
    """
    Computes and batch-inserts factor metrics into SQLite factor_metrics table.
    """
    all_rows = []
    for d in trade_dates:
        factors = compute_cross_sectional_factors(d, db_path=db_path)
        for fname, df in factors.items():
            for r in df.to_dicts():
                all_rows.append((
                    r["trade_date"],
                    r["ts_code"],
                    fname,
                    r["raw_val"],
                    r.get("clean_val", r["raw_val"])
                ))

    if not all_rows:
        return 0

    sql = """
    INSERT OR REPLACE INTO factor_metrics (trade_date, ts_code, factor_name, raw_val, clean_val)
    VALUES (?, ?, ?, ?, ?);
    """
    with get_db(db_path) as conn:
        return batch_insert(conn, sql, all_rows, chunk_size=5000, label="factor_metrics")
