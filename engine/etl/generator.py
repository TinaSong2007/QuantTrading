"""
High-Fidelity Synthetic A-Share Market Data Generator
Generates realistic CSI300 constituents, PIT financial statements with precalculated TTM,
and backward-adjusted daily bars including suspensions and limit-up/down events.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

SW_INDUSTRIES = [
    "银行", "非银金融", "电子", "计算机", "医药生物", "食品饮料",
    "电力设备", "机械设备", "汽车", "基础化工", "有色金属", "钢铁",
    "煤炭", "石油石化", "建筑装饰", "建筑材料", "通信", "传媒",
    "商贸零售", "社会服务", "家用电器", "纺织服饰", "轻工制造", "交通运输",
    "国防军工", "公用事业", "环保", "房地产", "农林牧渔", "美容护理", "综合"
]


def generate_trade_dates(start_date: str = "20220101", end_date: str = "20241231") -> List[str]:
    """Generates a list of valid weekday trading dates in YYYYMMDD format."""
    cur = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    trade_dates = []
    while cur <= end:
        if cur.weekday() < 5:  # Monday to Friday
            trade_dates.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return trade_dates


def generate_mock_stock_universe(n_stocks: int = 100) -> List[Dict]:
    """
    Generates representative A-share constituents across Main, ChiNext, and STAR boards.
    """
    stocks = []
    random.seed(42)
    for i in range(1, n_stocks + 1):
        if i <= int(n_stocks * 0.7):
            # Main board
            if i % 2 == 1:
                code = f"60{i:04d}.SH"
            else:
                code = f"00{i:04d}.SZ"
            board = "Main"
        elif i <= int(n_stocks * 0.88):
            # ChiNext
            code = f"30{i:04d}.SZ"
            board = "ChiNext"
        else:
            # STAR
            code = f"68{i:04d}.SH"
            board = "STAR"

        industry = SW_INDUSTRIES[(i - 1) % len(SW_INDUSTRIES)]
        is_st = 1 if i in [7, 33] else 0
        prefix = "ST" if is_st else "标的"
        name = f"{prefix}{i:03d}"
        list_year = 2010 + (i % 12)
        list_date = f"{list_year}{((i % 12) + 1):02d}15"

        stocks.append({
            "ts_code": code,
            "name": name,
            "industry": industry,
            "list_date": list_date,
            "is_st": is_st,
            "market_board": board,
        })
    return stocks


def generate_mock_daily_bars(
    stocks: List[Dict],
    trade_dates: List[str],
    initial_states: Optional[Dict[str, Tuple[float, float]]] = None
) -> List[Tuple]:
    """
    Generates realistic daily bars with backward adjustment factors, suspensions, and limit-up/down bars.
    Returns tuples matching daily_bar schema:
    (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close)
    """
    bars = []
    random.seed(2026)

    for stock in stocks:
        ts_code = stock["ts_code"]
        board = stock["market_board"]
        limit_pct = 0.20 if board in ("ChiNext", "STAR") else 0.10
        base_price = 20.0 + (hash(ts_code) % 80)
        
        if initial_states and ts_code in initial_states:
            cur_price, adj_factor = initial_states[ts_code]
        else:
            cur_price = base_price
            adj_factor = 1.0

        for idx, date in enumerate(trade_dates):
            # Skip if before listing date
            if date < stock["list_date"]:
                continue

            pre_close = round(cur_price, 2)

            # Occasional dividend adjustment event (every ~200 days)
            if idx > 0 and idx % 200 == 0:
                dividend = round(random.uniform(0.5, 1.5), 2)
                # Ex-dividend drops current price, factor increases
                ratio = pre_close / max(1.0, pre_close - dividend)
                adj_factor = round(adj_factor * ratio, 4)

            # 1. Check for suspension (vol = 0)
            is_suspended = (stock["is_st"] == 0 and random.random() < 0.015)
            if is_suspended:
                bars.append((
                    ts_code, date, pre_close, pre_close, pre_close, pre_close,
                    pre_close, 0.0, 0.0, adj_factor, round(pre_close * adj_factor, 4)
                ))
                continue

            # 2. Daily return simulation with fat tails
            ret = random.gauss(0.0003, 0.02)
            # Occasional limit-up or limit-down
            dice = random.random()
            if dice < 0.008:
                # One-word limit-up
                open_p = high_p = low_p = close_p = round(pre_close * (1 + limit_pct), 2)
            elif dice < 0.012:
                # One-word limit-down
                open_p = high_p = low_p = close_p = round(pre_close * (1 - limit_pct), 2)
            else:
                # Normal fluctuation
                bounded_ret = max(-limit_pct + 0.005, min(limit_pct - 0.005, ret))
                close_p = round(pre_close * (1 + bounded_ret), 2)
                day_range = abs(close_p - pre_close) + random.uniform(0.1, 0.5)
                high_p = round(max(pre_close, close_p) + random.uniform(0, day_range), 2)
                low_p = round(min(pre_close, close_p) - random.uniform(0, day_range), 2)
                low_p = max(0.5, low_p)
                open_p = round(pre_close * (1 + random.uniform(-0.01, 0.01)), 2)
                high_p = max(high_p, open_p, close_p)
                low_p = min(low_p, open_p, close_p)

            cur_price = close_p
            vol = round(random.uniform(20000, 500000), 0)
            amount = round(vol * ((open_p + close_p) / 2.0) * 100, 2)
            adj_close = round(close_p * adj_factor, 4)

            bars.append((
                ts_code, date, open_p, high_p, low_p, close_p,
                pre_close, vol, amount, adj_factor, adj_close
            ))

    return bars


def generate_mock_financial_pit(
    stocks: List[Dict],
    years: Optional[List[int]] = None,
    min_ann_date: Optional[str] = None
) -> List[Tuple]:
    """
    Generates quarterly PIT financial records with precomputed TTM net profit and ROE.
    Enforces strict disclosure timeline:
    Q1 (0331): Ann date between 0405 - 0430
    Q2 (0630): Ann date between 0715 - 0830
    Q3 (0930): Ann date between 1010 - 1030
    Q4 (1231): Ann date between next year 0120 - 0425
    Returns tuples matching financial_pit schema:
    (ts_code, ann_date, end_date, roe, net_profit, revenue, ttm_net_profit, roe_ttm)
    """
    if years is None:
        years = [2021, 2022, 2023, 2024, 2025, 2026]

    records = []
    random.seed(999)

    for stock in stocks:
        ts_code = stock["ts_code"]
        base_profit = random.uniform(5e7, 2e9)
        base_rev = base_profit * random.uniform(5.0, 15.0)

        # Track rolling 4 quarters for static TTM computation
        quarterly_profits: List[float] = []

        for yr in years:
            quarters = [
                ("0331", f"{yr}04{random.randint(10, 28):02d}", 0.25),
                ("0630", f"{yr}08{random.randint(10, 28):02d}", 0.50),
                ("0930", f"{yr}10{random.randint(15, 28):02d}", 0.75),
                ("1231", f"{yr+1}04{random.randint(10, 25):02d}", 1.00),
            ]

            for end_suffix, ann_date, cum_ratio in quarters:
                end_date = f"{yr}{end_suffix}"
                # Add random growth
                annual_factor = 1.0 + (yr - 2021) * 0.08 + random.uniform(-0.1, 0.15)
                q_net_profit = base_profit * annual_factor * 0.25 * random.uniform(0.85, 1.15)
                quarterly_profits.append(q_net_profit)

                # Cumulative numbers reported in China A-share interim reports
                if end_suffix == "0331":
                    cum_profit = q_net_profit
                elif end_suffix == "0630":
                    cum_profit = (quarterly_profits[-2] + quarterly_profits[-1]) if len(quarterly_profits) >= 2 else q_net_profit * 2.0
                elif end_suffix == "0930":
                    cum_profit = (quarterly_profits[-3] + quarterly_profits[-2] + quarterly_profits[-1]) if len(quarterly_profits) >= 3 else q_net_profit * 3.0
                else:
                    cum_profit = sum(quarterly_profits[-4:]) if len(quarterly_profits) >= 4 else q_net_profit * 4.0

                cum_rev = cum_profit * random.uniform(6.0, 12.0)
                roe = round(cum_profit / (base_profit * 6.0) * 100, 2)

                # Precomputed static rolling TTM: sum of last 4 quarters
                ttm_profit = sum(quarterly_profits[-4:]) if len(quarterly_profits) >= 4 else cum_profit * (4.0 / max(1, len(quarterly_profits)))
                roe_ttm = round(ttm_profit / (base_profit * 6.0) * 100, 2)

                # Only include records announced after min_ann_date if specified
                if min_ann_date and ann_date <= min_ann_date:
                    continue

                records.append((
                    ts_code,
                    ann_date,
                    end_date,
                    roe,
                    round(cum_profit, 2),
                    round(cum_rev, 2),
                    round(ttm_profit, 2),
                    roe_ttm
                ))

    return records
