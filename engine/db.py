"""
SQLite Warehouse Database Manager
Configured with WAL mode, 64MB cache, and 30-second lock timeout for high-concurrency quant pipelines.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "warehouse.db")


def get_connection(db_path: Optional[str] = None, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Returns a configured sqlite3 connection with WAL mode and high performance PRAGMAs.
    """
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row

    # Execute high-throughput PRAGMA configurations
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


@contextmanager
def get_db(db_path: Optional[str] = None, timeout: float = 30.0) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections ensuring proper commit/rollback and cleanup.
    """
    conn = get_connection(db_path, timeout=timeout)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initializes database schema with the 4 core institutional quant tables and performance indices.
    """
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # 1. Daily Bar Table (with backward adjustment support)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_bar (
            ts_code     TEXT NOT NULL,
            trade_date  TEXT NOT NULL,  -- YYYYMMDD
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            pre_close   REAL,
            vol         REAL,
            amount      REAL,
            adj_factor  REAL NOT NULL,  -- Cumulative backward adjust factor
            adj_close   REAL,           -- close * adj_factor
            PRIMARY KEY (ts_code, trade_date)
        );
        """)

        # 2. Point-in-Time (PIT) Financial Disclosures Table (with precomputed TTM)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_pit (
            ts_code         TEXT NOT NULL,
            ann_date        TEXT NOT NULL,  -- Announcement date (must be <= trade_date)
            end_date        TEXT NOT NULL,  -- Fiscal period end date (e.g. 20241231)
            roe             REAL,
            net_profit      REAL,
            revenue         REAL,
            ttm_net_profit  REAL,           -- Precomputed rolling TTM net profit
            roe_ttm         REAL,           -- Precomputed TTM ROE
            PRIMARY KEY (ts_code, ann_date, end_date)
        );
        """)

        # 3. Stock Basic & Industry Classification Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code         TEXT PRIMARY KEY,
            name            TEXT,
            industry        TEXT,          -- Shenwan Level-1 Industry
            list_date       TEXT,          -- Listing date YYYYMMDD
            is_st           INTEGER DEFAULT 0, -- 0: Normal, 1: ST / *ST
            market_board    TEXT DEFAULT 'Main' -- 'Main', 'ChiNext', 'STAR'
        );
        """)

        # 4. Factor Metrics & Analytics Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS factor_metrics (
            trade_date  TEXT NOT NULL,
            ts_code     TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            raw_val     REAL,
            clean_val   REAL,          -- MAD winzorized + Z-scored + OLS neutralized residual
            PRIMARY KEY (trade_date, ts_code, factor_name)
        );
        """)

        # Performance Indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bar_date ON daily_bar(trade_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bar_code ON daily_bar(ts_code);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pit_ann ON financial_pit(ann_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pit_code ON financial_pit(ts_code);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_name ON factor_metrics(factor_name, trade_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_date ON factor_metrics(trade_date);")


def get_latest_trade_date(db_path: Optional[str] = None) -> Optional[str]:
    """Returns the latest trade_date recorded in daily_bar, or None if empty."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(trade_date) FROM daily_bar;")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def get_earliest_trade_date(db_path: Optional[str] = None) -> Optional[str]:
    """Returns the earliest trade_date recorded in daily_bar, or None if empty."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(trade_date) FROM daily_bar;")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def get_stock_latest_states(db_path: Optional[str] = None) -> dict[str, tuple[str, float, float]]:
    """
    Returns the latest state for each stock: ts_code -> (latest_trade_date, close, adj_factor).
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    sql = """
    WITH RankedBars AS (
        SELECT
            ts_code,
            trade_date,
            close,
            adj_factor,
            ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rnk
        FROM daily_bar
    )
    SELECT ts_code, trade_date, close, adj_factor
    FROM RankedBars
    WHERE rnk = 1;
    """
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()
    return {r["ts_code"]: (r["trade_date"], float(r["close"] or 10.0), float(r["adj_factor"] or 1.0)) for r in rows}


def get_latest_financial_ann_date(db_path: Optional[str] = None) -> Optional[str]:
    """Returns the maximum announcement date (ann_date) recorded in financial_pit."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(ann_date) FROM financial_pit;")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


if __name__ == "__main__":
    init_db()
    print("Database schema successfully initialized.")

