"""
Tests for SQLite WAL Configuration, Schema Constraints, and Concurrency Protections.
"""

import os
import sqlite3
import tempfile
import pytest

from engine.db import get_connection, get_db, init_db


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)
    wal_file = f"{db_path}-wal"
    shm_file = f"{db_path}-shm"
    if os.path.exists(wal_file):
        os.remove(wal_file)
    if os.path.exists(shm_file):
        os.remove(shm_file)


def test_wal_pragmas_and_schema(temp_db):
    conn = get_connection(temp_db, timeout=30.0)
    cursor = conn.cursor()

    # 1. Verify WAL Mode
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

    # 2. Verify Synchronous Mode (1 = NORMAL)
    cursor.execute("PRAGMA synchronous;")
    sync = cursor.fetchone()[0]
    assert sync == 1, f"Expected synchronous NORMAL (1), got {sync}"

    # 3. Verify 4 Core Tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    expected = {"daily_bar", "financial_pit", "stock_basic", "factor_metrics"}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    conn.close()


def test_composite_primary_key_enforcement(temp_db):
    with get_db(temp_db) as conn:
        cursor = conn.cursor()
        # Insert a daily bar
        cursor.execute("""
            INSERT INTO daily_bar (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close)
            VALUES ('000001.SZ', '20240102', 10.0, 10.5, 9.9, 10.2, 10.0, 10000, 102000, 1.25, 12.75);
        """)

        # Attempting identical PK insert should raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO daily_bar (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close)
                VALUES ('000001.SZ', '20240102', 10.1, 10.6, 9.8, 10.3, 10.0, 12000, 123000, 1.25, 12.875);
            """)

        # INSERT OR REPLACE should update cleanly
        cursor.execute("""
            INSERT OR REPLACE INTO daily_bar (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close)
            VALUES ('000001.SZ', '20240102', 10.1, 10.6, 9.8, 10.3, 10.0, 12000, 123000, 1.25, 12.875);
        """)

        cursor.execute("SELECT close, adj_close FROM daily_bar WHERE ts_code='000001.SZ' AND trade_date='20240102'")
        row = cursor.fetchone()
        assert row["close"] == 10.3
        assert row["adj_close"] == 12.875
