"""
Institutional Factor Evaluation and Stratified Quantile Backtest Engine.
Features:
1. Anti-lookahead return alignment (T+1 -> T+1+L with backward-adjusted prices).
2. Rank IC (Spearman), Cumulative IC, IC-IR for arbitrary lags (1, 5, 20).
3. 10-Tier Quantile Layered Backtest with 0.1% transaction friction.
4. Non-tradable carry-over weight rebalancing for suspended & limit-down assets.
5. Turnover monitoring (per-period & annualized) and monthly heatmap matrix.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from engine.db import get_connection
from engine.pit_query import get_tradable_mask


def fetch_factor_and_prices(
    factor_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_clean_val: bool = True,
    db_path: Optional[str] = None
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Fetches factor metrics and daily backward-adjusted prices for backtesting.
    """
    conn = get_connection(db_path)
    val_col = "clean_val" if use_clean_val else "raw_val"

    date_filter = ""
    params: List[Any] = [factor_name]
    if start_date:
        date_filter += " AND trade_date >= ?"
        params.append(start_date)
    if end_date:
        date_filter += " AND trade_date <= ?"
        params.append(end_date)

    factor_sql = f"""
        SELECT trade_date, ts_code, {val_col} as factor_val
        FROM factor_metrics
        WHERE factor_name = ? {date_filter}
        ORDER BY trade_date ASC;
    """
    cursor = conn.cursor()
    cursor.execute(factor_sql, params)
    f_rows = cursor.fetchall()

    price_sql = """
        SELECT ts_code, trade_date, open, high, low, close, pre_close, vol, adj_factor, adj_close
        FROM daily_bar
        ORDER BY trade_date ASC;
    """
    cursor.execute(price_sql)
    p_rows = cursor.fetchall()
    conn.close()

    f_df = pl.DataFrame([dict(r) for r in f_rows]) if f_rows else pl.DataFrame()
    p_df = pl.DataFrame([dict(r) for r in p_rows]) if p_rows else pl.DataFrame()

    return f_df, p_df


def calculate_rank_ic(
    factor_name: str,
    lags: List[int] = [1, 5, 20],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_clean_val: bool = True,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculates multi-lag Spearman Rank IC series, Cumulative IC, and IC-IR.
    Strictly aligns factor at date T with backward-adjusted forward returns from T+1 to T+1+lag.
    """
    f_df, p_df = fetch_factor_and_prices(
        factor_name, start_date=start_date, end_date=end_date, use_clean_val=use_clean_val, db_path=db_path
    )

    if f_df.is_empty() or p_df.is_empty():
        return {}

    # Get chronological trade dates
    unique_dates = sorted(p_df["trade_date"].unique().to_list())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    # Build fast price lookup: (trade_date, ts_code) -> adj_close
    p_dicts = p_df.select(["trade_date", "ts_code", "adj_close"]).to_dicts()
    price_map = {(r["trade_date"], r["ts_code"]): r["adj_close"] for r in p_dicts}

    factor_by_date = defaultdict(list)
    for row in f_df.to_dicts():
        factor_by_date[row["trade_date"]].append(row)

    results: Dict[str, Any] = {
        "lags": lags,
        "ic_series": {lag: [] for lag in lags},
        "dates": {lag: [] for lag in lags},
        "summary": {},
    }

    for lag in lags:
        for t_date in sorted(factor_by_date.keys()):
            if t_date not in date_to_idx:
                continue
            idx_t = date_to_idx[t_date]

            # Next day T+1 for entry, and T+1+lag for exit
            idx_entry = idx_t + 1
            idx_exit = idx_t + 1 + lag
            if idx_exit >= len(unique_dates):
                continue

            entry_date = unique_dates[idx_entry]
            exit_date = unique_dates[idx_exit]

            f_items = factor_by_date[t_date]
            factor_vals = []
            fwd_returns = []

            for item in f_items:
                code = item["ts_code"]
                f_val = item["factor_val"]
                if f_val is None:
                    continue

                p_entry = price_map.get((entry_date, code))
                p_exit = price_map.get((exit_date, code))

                if p_entry and p_exit and p_entry > 0:
                    ret = (p_exit - p_entry) / p_entry
                    factor_vals.append(f_val)
                    fwd_returns.append(ret)

            if len(factor_vals) >= 10:
                spearman_corr, _ = stats.spearmanr(factor_vals, fwd_returns)
                if not np.isnan(spearman_corr):
                    results["ic_series"][lag].append(float(spearman_corr))
                    results["dates"][lag].append(t_date)

        # Summary statistics for this lag
        ic_arr = np.array(results["ic_series"][lag])
        if len(ic_arr) > 0:
            ic_mean = float(np.mean(ic_arr))
            ic_std = float(np.std(ic_arr))
            ic_ir = float((ic_mean / ic_std) * np.sqrt(252 / lag)) if ic_std > 1e-6 else 0.0
            pos_ratio = float(np.mean(ic_arr > 0))
            cum_ic = list(np.cumsum(ic_arr))

            results["summary"][lag] = {
                "ic_mean": round(ic_mean, 4),
                "ic_std": round(ic_std, 4),
                "ic_ir": round(ic_ir, 3),
                "positive_ratio": round(pos_ratio * 100, 2),
                "n_samples": len(ic_arr),
                "cum_ic": cum_ic,
            }

    return results


def run_quantile_backtest(
    factor_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    n_groups: int = 10,
    rebalance_interval: int = 5,  # 5 days = weekly
    fee_rate: float = 0.001,      # 0.1% single-trip transaction cost
    use_clean_val: bool = True,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a 10-tier quantile layered backtest with:
    - 0.1% single-trip transaction friction.
    - Suspended and limit-down assets carry-over rebalancing.
    - Turnover rate monitoring (per-period and annualized).
    - Monotonicity score and monthly Long-Short return heatmap.
    """
    f_df, p_df = fetch_factor_and_prices(
        factor_name, start_date=start_date, end_date=end_date, use_clean_val=use_clean_val, db_path=db_path
    )

    if f_df.is_empty() or p_df.is_empty():
        return {}

    all_trade_dates = sorted(p_df["trade_date"].unique().to_list())
    date_to_idx = {d: i for i, d in enumerate(all_trade_dates)}

    # Build price lookup: (trade_date, ts_code) -> dict
    p_dicts = p_df.to_dicts()
    bar_map = {(r["trade_date"], r["ts_code"]): r for r in p_dicts}

    factor_by_date = defaultdict(list)
    for row in f_df.to_dicts():
        factor_by_date[row["trade_date"]].append(row)

    # Rebalance schedule
    factor_dates = sorted(factor_by_date.keys())
    # Subsample rebalance dates according to interval
    rebalance_dates = factor_dates[::rebalance_interval]
    if len(rebalance_dates) < 3:
        rebalance_dates = factor_dates

    # NAV tracking: group_idx (1..n_groups) -> list of NAVs
    nav_history: Dict[int, List[float]] = {g: [1.0] for g in range(1, n_groups + 1)}
    benchmark_nav: List[float] = [1.0]
    nav_dates: List[str] = []

    # Current holdings: group_idx -> dict {ts_code: weight}
    current_weights: Dict[int, Dict[str, float]] = {g: {} for g in range(1, n_groups + 1)}

    turnover_records: Dict[int, List[float]] = {g: [] for g in range(1, n_groups + 1)}
    rebal_periods: List[str] = []

    for t_idx, t_date in enumerate(rebalance_dates[:-1]):
        next_t_date = rebalance_dates[t_idx + 1]
        cur_bar_idx = date_to_idx.get(t_date)
        next_bar_idx = date_to_idx.get(next_t_date)
        if cur_bar_idx is None or next_bar_idx is None:
            continue

        # Execution entry date is T+1, exit date is Next_T+1
        entry_idx = cur_bar_idx + 1
        exit_idx = next_bar_idx + 1
        if exit_idx >= len(all_trade_dates):
            continue

        entry_date = all_trade_dates[entry_idx]
        exit_date = all_trade_dates[exit_idx]
        rebal_periods.append(entry_date)

        # Get dynamic tradable mask on T and T+1
        tradable_df = get_tradable_mask(
            trade_date=t_date, next_trade_date=entry_date, db_path=db_path
        )
        tradable_map = {r["ts_code"]: r for r in tradable_df.to_dicts()}

        f_items = factor_by_date[t_date]
        # Filter items where can_buy is True or is_tradable is True
        valid_items = []
        for item in f_items:
            code = item["ts_code"]
            t_info = tradable_map.get(code)
            if t_info and t_info["is_tradable"] and item["factor_val"] is not None:
                valid_items.append(item)

        if len(valid_items) < n_groups:
            continue

        # Sort by factor value ascending
        valid_items.sort(key=lambda x: x["factor_val"])
        group_size = len(valid_items) // n_groups

        # All market return (benchmark proxy)
        market_rets = []

        # Process each quantile group
        for g in range(1, n_groups + 1):
            if g == n_groups:
                g_items = valid_items[(g - 1) * group_size :]
            else:
                g_items = valid_items[(g - 1) * group_size : g * group_size]

            target_codes = {item["ts_code"] for item in g_items}
            old_weights = current_weights[g]

            # EDGE CASE: Carry-over non-tradable assets
            # If an old asset cannot be sold on entry_date (limit-down or suspended), carry over weight
            carried_weights = {}
            for code, w in old_weights.items():
                if code not in target_codes:
                    t_info = tradable_map.get(code)
                    # Cannot sell? Carry over!
                    if t_info and not t_info["can_sell"]:
                        carried_weights[code] = w

            carried_sum = sum(carried_weights.values())
            remaining_weight = max(0.0, 1.0 - carried_sum)

            # Buyable targets on entry_date
            buyable_targets = []
            for code in target_codes:
                t_info = tradable_map.get(code)
                if t_info and t_info["can_buy"]:
                    buyable_targets.append(code)

            new_weights = dict(carried_weights)
            if buyable_targets and remaining_weight > 0:
                each_w = remaining_weight / len(buyable_targets)
                for code in buyable_targets:
                    new_weights[code] = new_weights.get(code, 0.0) + each_w
            elif not carried_weights and target_codes:
                # Fallback equal weight
                each_w = 1.0 / len(target_codes)
                for code in target_codes:
                    new_weights[code] = each_w

            # Compute turnover: 0.5 * sum(|w_new - w_old|)
            all_codes = set(old_weights.keys()).union(set(new_weights.keys()))
            turnover = 0.5 * sum(abs(new_weights.get(c, 0.0) - old_weights.get(c, 0.0)) for c in all_codes)
            turnover_records[g].append(turnover)

            # Compute forward portfolio return from entry_date to exit_date
            period_ret = 0.0
            for code, w in new_weights.items():
                b_entry = bar_map.get((entry_date, code))
                b_exit = bar_map.get((exit_date, code))
                if b_entry and b_exit and b_entry["adj_close"] > 0:
                    asset_ret = (b_exit["adj_close"] - b_entry["adj_close"]) / b_entry["adj_close"]
                    period_ret += w * asset_ret
                    market_rets.append(asset_ret)

            # Deduct transaction friction: fee_rate * turnover
            period_cost = fee_rate * turnover
            net_ret = period_ret - period_cost

            # Update NAV
            prev_nav = nav_history[g][-1]
            new_nav = prev_nav * (1.0 + net_ret)
            nav_history[g].append(new_nav)
            current_weights[g] = new_weights

        # Benchmark equal-weighted return
        b_ret = float(np.mean(market_rets)) if market_rets else 0.0
        benchmark_nav.append(benchmark_nav[-1] * (1.0 + b_ret))
        nav_dates.append(exit_date)

    # Calculate Long-Short (Top - Bottom) spread NAV
    q_top = nav_history[n_groups]
    q_bot = nav_history[1]
    long_short_nav = [t / b for t, b in zip(q_top, q_bot)]

    # Performance metrics calculation
    annual_factor = 252 / rebalance_interval
    metrics = {}
    for g in range(1, n_groups + 1):
        nav_arr = np.array(nav_history[g])
        rets = np.diff(nav_arr) / nav_arr[:-1]
        ann_ret = float((nav_arr[-1] ** (annual_factor / max(1, len(rets)))) - 1.0) if len(rets) > 0 else 0.0
        ann_vol = float(np.std(rets) * np.sqrt(annual_factor)) if len(rets) > 0 else 0.0
        sharpe = round((ann_ret - 0.02) / ann_vol, 3) if ann_vol > 1e-6 else 0.0

        # Maximum Drawdown
        cum_max = np.maximum.accumulate(nav_arr)
        drawdowns = (nav_arr - cum_max) / cum_max
        mdd = round(float(np.min(drawdowns)), 4)
        calmar = round(abs(ann_ret / mdd), 3) if abs(mdd) > 1e-6 else 0.0

        # Annualized turnover
        avg_turnover = float(np.mean(turnover_records[g])) if turnover_records[g] else 0.0
        ann_turnover = round(avg_turnover * annual_factor, 2)

        metrics[f"Group_{g}"] = {
            "annual_return": round(ann_ret * 100, 2),
            "annual_volatility": round(ann_vol * 100, 2),
            "sharpe_ratio": sharpe,
            "max_drawdown": round(mdd * 100, 2),
            "calmar_ratio": calmar,
            "annual_turnover": ann_turnover,
            "final_nav": round(float(nav_arr[-1]), 4),
        }

    # Long-Short metrics
    ls_arr = np.array(long_short_nav)
    ls_rets = np.diff(ls_arr) / ls_arr[:-1]
    ls_ann_ret = float((ls_arr[-1] ** (annual_factor / max(1, len(ls_rets)))) - 1.0) if len(ls_rets) > 0 else 0.0
    ls_ann_vol = float(np.std(ls_rets) * np.sqrt(annual_factor)) if len(ls_rets) > 0 else 0.0
    ls_sharpe = round(ls_ann_ret / ls_ann_vol, 3) if ls_ann_vol > 1e-6 else 0.0
    ls_cum_max = np.maximum.accumulate(ls_arr)
    ls_mdd = round(float(np.min((ls_arr - ls_cum_max) / ls_cum_max)), 4)

    metrics["Long_Short"] = {
        "annual_return": round(ls_ann_ret * 100, 2),
        "annual_volatility": round(ls_ann_vol * 100, 2),
        "sharpe_ratio": ls_sharpe,
        "max_drawdown": round(ls_mdd * 100, 2),
        "calmar_ratio": round(abs(ls_ann_ret / ls_mdd), 3) if abs(ls_mdd) > 1e-6 else 0.0,
        "final_nav": round(float(ls_arr[-1]), 4),
    }

    # Monotonicity test: Rank correlation of group IDs (1..10) vs their final returns
    group_returns = [metrics[f"Group_{g}"]["final_nav"] for g in range(1, n_groups + 1)]
    monotonicity_corr, _ = stats.spearmanr(list(range(1, n_groups + 1)), group_returns)

    # Monthly return heatmap for Long-Short spread
    heatmap_data = build_monthly_heatmap(nav_dates, ls_rets)

    return {
        "dates": nav_dates,
        "nav_history": nav_history,
        "long_short_nav": long_short_nav,
        "benchmark_nav": benchmark_nav,
        "metrics": metrics,
        "monotonicity_score": round(float(monotonicity_corr), 3),
        "monthly_heatmap": heatmap_data,
        "turnover_records": turnover_records,
    }


def build_monthly_heatmap(dates: List[str], rets: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Aggregates periodic returns into Year x Month percentage returns.
    """
    if len(dates) == 0 or len(rets) == 0:
        return {}

    df = pd.DataFrame({
        "date": [pd.to_datetime(d, format="%Y%m%d") for d in dates[:len(rets)]],
        "ret": rets
    })
    df["year"] = df["date"].dt.year.astype(str)
    df["month"] = df["date"].dt.strftime("%m")

    # Compound monthly return: prod(1 + r) - 1
    monthly = df.groupby(["year", "month"])["ret"].apply(lambda s: (np.prod(1.0 + s) - 1.0) * 100.0).reset_index()

    heatmap: Dict[str, Dict[str, float]] = defaultdict(dict)
    for _, row in monthly.iterrows():
        heatmap[str(row["year"])][str(row["month"])] = round(float(row["ret"]), 2)

    return dict(heatmap)
