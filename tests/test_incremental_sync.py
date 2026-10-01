"""
Unit Tests for 5-Year Data Horizon and Incremental Breakpoint Sync Mechanism.
"""

import os
import tempfile
import pytest

from engine.db import get_connection, get_earliest_trade_date, get_latest_trade_date, init_db
from engine.etl.collector import auto_sync_on_startup, sync_market_data
from engine.etl.generator import generate_mock_daily_bars, generate_mock_stock_universe


@pytest.fixture
def fresh_test_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    f.close()
    init_db(db_path)

    yield db_path

    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in ["-wal", "-shm"]:
        if os.path.exists(f"{db_path}{ext}"):
            os.remove(f"{db_path}{ext}")


def test_full_5_year_sync_on_empty_db(fresh_test_db):
    # Run sync on an empty database
    res = sync_market_data(years=5, n_stocks=10, db_path=fresh_test_db, calculate_factors=False)

    assert res["status"] == "full_sync"
    assert res["bars_added"] > 0

    latest = get_latest_trade_date(fresh_test_db)
    earliest = get_earliest_trade_date(fresh_test_db)
    assert latest is not None
    assert earliest is not None
    # 5 years difference between earliest and latest
    yr_diff = int(latest[:4]) - int(earliest[:4])
    assert yr_diff >= 4


def test_incremental_breakpoint_sync(fresh_test_db):
    # Step 1: Initialize database with data up to 20240105
    stocks = generate_mock_stock_universe(10)
    from engine.etl.collector import ingest_daily_bars, ingest_stock_basics
    ingest_stock_basics(stocks, fresh_test_db)

    initial_dates = ["20240102", "20240103", "20240104", "20240105"]
    initial_bars = generate_mock_daily_bars(stocks, initial_dates)
    ingest_daily_bars(initial_bars, fresh_test_db)

    assert get_latest_trade_date(fresh_test_db) == "20240105"

    conn = get_connection(fresh_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM daily_bar;")
    initial_count = cursor.fetchone()[0]
    conn.close()

    # Step 2: Trigger incremental sync
    res = sync_market_data(years=5, n_stocks=10, db_path=fresh_test_db, calculate_factors=False)

    assert res["status"] == "updated"
    assert res["bars_added"] > 0

    latest_after = get_latest_trade_date(fresh_test_db)
    assert latest_after > "20240105"

    conn = get_connection(fresh_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM daily_bar;")
    final_count = cursor.fetchone()[0]
    conn.close()
    assert final_count > initial_count

    # Step 3: Trigger sync again immediately -> should recognize it is already up to date
    res_repeat = sync_market_data(years=5, n_stocks=10, db_path=fresh_test_db, calculate_factors=False)
    assert res_repeat["status"] == "up_to_date"
    assert res_repeat["bars_added"] == 0
