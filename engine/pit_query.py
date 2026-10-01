"""
Point-in-Time (PIT) Financial Query and Dynamic Tradable Mask Engine.
Strictly prevents lookahead bias (ann_date <= trade_date) and enforces trading constraints
(ST, new listings <120d, suspensions, and one-word limit-up/down execution blocks).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import polars as pl

from engine.db import get_connection


def get_pit_financials(
    trade_date: str,
    db_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Returns the latest known quarterly financial report for each stock as of trade_date T.
    Strict Condition: ann_date <= trade_date.
    If multiple reports exist up to trade_date, takes the most recent announcement date,
    and latest fiscal period end_date. Reads precomputed rolling TTM numbers directly.
    """
    conn = get_connection(db_path)
    # Using window function ROW_NUMBER to extract the single latest disclosure per ts_code
    sql = """
    WITH RankedFinancials AS (
        SELECT
            ts_code,
            ann_date,
            end_date,
            roe,
            net_profit,
            revenue,
            ttm_net_profit,
            roe_ttm,
            ROW_NUMBER() OVER (
                PARTITION BY ts_code
                ORDER BY ann_date DESC, end_date DESC
            ) as rnk
        FROM financial_pit
        WHERE ann_date <= ?
    )
    SELECT
        ts_code,
        ann_date,
        end_date,
        roe,
        net_profit,
        revenue,
        ttm_net_profit,
        roe_ttm
    FROM RankedFinancials
    WHERE rnk = 1;
    """
    cursor = conn.cursor()
    cursor.execute(sql, (trade_date,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return pl.DataFrame(schema={
            "ts_code": pl.String,
            "ann_date": pl.String,
            "end_date": pl.String,
            "roe": pl.Float64,
            "net_profit": pl.Float64,
            "revenue": pl.Float64,
            "ttm_net_profit": pl.Float64,
            "roe_ttm": pl.Float64,
        })

    data = [dict(r) for r in rows]
    return pl.DataFrame(data)


def get_tradable_mask(
    trade_date: str,
    next_trade_date: Optional[str] = None,
    min_list_days: int = 120,
    db_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Evaluates dynamic tradable status for all stocks as of trade_date T (and execution on next_trade_date T+1).
    Filters:
      1. ST / *ST status (is_st == 1 or 'ST' in name)
      2. New listing: listed for less than min_list_days
      3. Suspended on trade_date (vol == 0)
      4. One-word limit-up on next_trade_date (open == high and return >= 9.8% / 19.8%): CANNOT BUY
      5. One-word limit-down on next_trade_date (open == low and return <= -9.8% / -19.8%): CANNOT SELL
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # 1. Fetch stock basic info
    cursor.execute("SELECT ts_code, name, industry, list_date, is_st, market_board FROM stock_basic;")
    basics = {r["ts_code"]: dict(r) for r in cursor.fetchall()}

    # 2. Fetch daily bar on trade_date T
    cursor.execute("""
        SELECT ts_code, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close
        FROM daily_bar
        WHERE trade_date = ?;
    """, (trade_date,))
    t_bars = {r["ts_code"]: dict(r) for r in cursor.fetchall()}

    # 3. Fetch next day bar on T+1 (if provided) to detect execution locks
    next_bars = {}
    if next_trade_date:
        cursor.execute("""
            SELECT ts_code, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close
            FROM daily_bar
            WHERE trade_date = ?;
        """, (next_trade_date,))
        next_bars = {r["ts_code"]: dict(r) for r in cursor.fetchall()}

    conn.close()

    t_dt = datetime.strptime(trade_date, "%Y%m%d")

    records = []
    for ts_code, basic in basics.items():
        name = basic["name"] or ""
        is_st = bool(basic["is_st"] or "ST" in name)
        board = basic["market_board"]
        limit_threshold = 0.198 if board in ("ChiNext", "STAR") else 0.098

        # Listing age check
        list_dt = datetime.strptime(basic["list_date"], "%Y%m%d")
        days_listed = (t_dt - list_dt).days
        is_new = days_listed < min_list_days

        # Suspension check on T
        bar_t = t_bars.get(ts_code)
        is_suspended_t = (bar_t is None or bar_t["vol"] is None or bar_t["vol"] <= 0)

        # Execution checks on T+1
        is_suspended_next = False
        is_limit_up_next = False
        is_limit_down_next = False

        if next_trade_date:
            bar_next = next_bars.get(ts_code)
            if bar_next is None or bar_next["vol"] is None or bar_next["vol"] <= 0:
                is_suspended_next = True
            else:
                o = bar_next["open"]
                h = bar_next["high"]
                l = bar_next["low"]
                pre_c = bar_next["pre_close"]
                if pre_c and pre_c > 0:
                    ret_open = (o - pre_c) / pre_c
                    # One-word limit-up: open == high and ret >= threshold
                    if o == h and ret_open >= limit_threshold:
                        is_limit_up_next = True
                    # One-word limit-down: open == low and ret <= -threshold
                    if o == l and ret_open <= -limit_threshold:
                        is_limit_down_next = True

        # Base tradable condition on T
        is_tradable = (not is_st) and (not is_new) and (not is_suspended_t)

        # Execution feasibility on T+1
        can_buy = is_tradable and (not is_suspended_next) and (not is_limit_up_next)
        can_sell = (not is_suspended_next) and (not is_limit_down_next)

        reasons = []
        if is_st:
            reasons.append("ST")
        if is_new:
            reasons.append("NewStock")
        if is_suspended_t:
            reasons.append("Suspended_T")
        if is_limit_up_next:
            reasons.append("LimitUp_BuyBlocked")
        if is_limit_down_next:
            reasons.append("LimitDown_SellBlocked")

        records.append({
            "ts_code": ts_code,
            "trade_date": trade_date,
            "industry": basic["industry"],
            "is_tradable": is_tradable,
            "can_buy": can_buy,
            "can_sell": can_sell,
            "is_st": is_st,
            "is_new": is_new,
            "is_suspended_t": is_suspended_t,
            "is_limit_up_next": is_limit_up_next,
            "is_limit_down_next": is_limit_down_next,
            "filter_reason": ";".join(reasons) if reasons else "OK",
        })

    return pl.DataFrame(records)
