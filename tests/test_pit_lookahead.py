"""
Unit Tests for Anti-Lookahead Bias in PIT Financial Disclosures & Execution Trading Mask.
"""

import os
import tempfile
import polars as pl
import pytest

from engine.db import get_db, init_db
from engine.pit_query import get_pit_financials, get_tradable_mask


@pytest.fixture
def mock_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        # Stock basic setup
        cursor.execute("""
            INSERT INTO stock_basic (ts_code, name, industry, list_date, is_st, market_board)
            VALUES
                ('600000.SH', '浦发银行', '银行', '20100101', 0, 'Main'),
                ('000001.SZ', '平安银行', '银行', '20100101', 0, 'Main'),
                ('000002.SZ', 'ST万科', '房地产', '20100101', 1, 'Main'),
                ('300001.SZ', '特锐德', '电力设备', '20240101', 0, 'ChiNext');
        """)

        # Financial PIT setup for 600000.SH
        # Q3 2024 report disclosed on 20241028
        cursor.execute("""
            INSERT INTO financial_pit (ts_code, ann_date, end_date, roe, net_profit, revenue, ttm_net_profit, roe_ttm)
            VALUES ('600000.SH', '20241028', '20240930', 8.5, 4.5e10, 1.2e11, 6.0e10, 11.2);
        """)
        # 2024 Annual report disclosed on 20250420
        cursor.execute("""
            INSERT INTO financial_pit (ts_code, ann_date, end_date, roe, net_profit, revenue, ttm_net_profit, roe_ttm)
            VALUES ('600000.SH', '20250420', '20241231', 10.2, 5.8e10, 1.6e11, 5.8e10, 10.2);
        """)

        # Daily bar for T (20250401) and T+1 (20250402)
        cursor.execute("""
            INSERT INTO daily_bar (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close)
            VALUES
                ('600000.SH', '20250401', 10.0, 10.2, 9.9, 10.1, 10.0, 50000, 505000, 1.0, 10.1),
                ('600000.SH', '20250402', 10.1, 10.3, 10.0, 10.2, 10.1, 60000, 612000, 1.0, 10.2),
                -- 000001.SZ hits one-word limit-up on T+1 (open == high, +10%)
                ('000001.SZ', '20250401', 12.0, 12.2, 11.9, 12.0, 12.0, 30000, 360000, 1.0, 12.0),
                ('000001.SZ', '20250402', 13.2, 13.2, 13.2, 13.2, 12.0, 1000, 13200, 1.0, 13.2),
                -- 000002.SZ is suspended on T
                ('000002.SZ', '20250401', 8.0, 8.0, 8.0, 8.0, 8.0, 0, 0, 1.0, 8.0);
        """)

    yield db_path

    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in ["-wal", "-shm"]:
        if os.path.exists(f"{db_path}{ext}"):
            os.remove(f"{db_path}{ext}")


def test_anti_lookahead_financial_pit(mock_db):
    # Query on 2025-04-01 (before 2025-04-20 annual report announcement)
    df_april_1 = get_pit_financials("20250401", db_path=mock_db)
    row_april_1 = df_april_1.filter(pl.col("ts_code") == "600000.SH")
    assert len(row_april_1) == 1
    # MUST see Q3 2024 report (end_date: 20240930, ann_date: 20241028)
    assert row_april_1["ann_date"][0] == "20241028"
    assert row_april_1["end_date"][0] == "20240930"
    assert row_april_1["net_profit"][0] == 4.5e10

    # Query on 2025-04-21 (after announcement date)
    df_april_21 = get_pit_financials("20250421", db_path=mock_db)
    row_april_21 = df_april_21.filter(pl.col("ts_code") == "600000.SH")
    assert len(row_april_21) == 1
    # Now sees the 2024 annual report
    assert row_april_21["ann_date"][0] == "20250420"
    assert row_april_21["end_date"][0] == "20241231"
    assert row_april_21["net_profit"][0] == 5.8e10


def test_dynamic_tradable_mask_and_limit_up(mock_db):
    mask = get_tradable_mask(
        trade_date="20250401",
        next_trade_date="20250402",
        min_list_days=120,
        db_path=mock_db
    )

    # 1. 600000.SH is completely normal and tradable
    pf = mask.filter(pl.col("ts_code") == "600000.SH")
    assert pf["is_tradable"][0] is True
    assert pf["can_buy"][0] is True
    assert pf["can_sell"][0] is True

    # 2. 000002.SZ is ST and Suspended on T
    st_stock = mask.filter(pl.col("ts_code") == "000002.SZ")
    assert st_stock["is_tradable"][0] is False
    assert st_stock["is_st"][0] is True
    assert st_stock["is_suspended_t"][0] is True

    # 3. 000001.SZ is tradable on T, but hits one-word limit-up on T+1 -> cannot buy
    limit_stock = mask.filter(pl.col("ts_code") == "000001.SZ")
    assert limit_stock["is_tradable"][0] is True
    assert limit_stock["is_limit_up_next"][0] is True
    assert limit_stock["can_buy"][0] is False  # Cannot buy!
