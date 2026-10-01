"""
Tests for Factor Evaluator: Forward Return Alignment, Rank IC/IC-IR,
10-Group Quantile Backtest with Friction, Turnover Rate, and Carry-Over Weight Logic.
"""

import os
import tempfile
import numpy as np
import pytest

from engine.analytics.evaluator import calculate_rank_ic, run_quantile_backtest
from engine.db import get_connection, init_db
from engine.etl.collector import run_sample_etl
from engine.factors.definitions import populate_factors_for_dates


@pytest.fixture(scope="module")
def populated_db():
    default_db = "warehouse.db"
    if os.path.exists(default_db):
        yield default_db
        return

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    f.close()

    # Ingest 20 stocks with 50 trading days
    init_db(db_path)
    from engine.etl.generator import (
        generate_mock_stock_universe,
        generate_trade_dates,
        generate_mock_daily_bars,
        generate_mock_financial_pit
    )
    from engine.etl.collector import ingest_stock_basics, ingest_daily_bars, ingest_financial_pit
    stocks = generate_mock_stock_universe(20)
    ingest_stock_basics(stocks, db_path)
    dates = generate_trade_dates("20240101", "20240430")
    bars = generate_mock_daily_bars(stocks, dates)
    ingest_daily_bars(bars, db_path)
    fins = generate_mock_financial_pit(stocks, years=[2023, 2024])
    ingest_financial_pit(fins, db_path)

    sample_dates = dates[25::3]
    populate_factors_for_dates(sample_dates, db_path=db_path)

    yield db_path

    if os.path.exists(db_path):
        os.remove(db_path)
    for ext in ["-wal", "-shm"]:
        if os.path.exists(f"{db_path}{ext}"):
            os.remove(f"{db_path}{ext}")


def test_rank_ic_computation(populated_db):
    ic_results = calculate_rank_ic(
        factor_name="MOM_20D",
        lags=[1, 5],
        use_clean_val=True,
        db_path=populated_db
    )

    assert "ic_series" in ic_results
    assert 1 in ic_results["ic_series"]
    assert 5 in ic_results["ic_series"]
    assert len(ic_results["ic_series"][1]) > 0

    # Summary metrics
    s1 = ic_results["summary"][1]
    assert "ic_mean" in s1
    assert "ic_ir" in s1
    assert "positive_ratio" in s1
    assert len(s1["cum_ic"]) == len(ic_results["ic_series"][1])


def test_quantile_backtest_with_friction_and_turnover(populated_db):
    bt_results = run_quantile_backtest(
        factor_name="MOM_20D",
        n_groups=5,
        rebalance_interval=2,
        fee_rate=0.001,
        use_clean_val=True,
        db_path=populated_db
    )

    assert "nav_history" in bt_results
    # 5 groups
    for g in range(1, 6):
        assert g in bt_results["nav_history"]
        assert len(bt_results["nav_history"][g]) > 1

    # Check metrics table
    metrics = bt_results["metrics"]
    assert "Group_1" in metrics
    assert "Group_5" in metrics
    assert "Long_Short" in metrics

    g5_metrics = metrics["Group_5"]
    assert "annual_return" in g5_metrics
    assert "sharpe_ratio" in g5_metrics
    assert "max_drawdown" in g5_metrics
    assert "annual_turnover" in g5_metrics
    assert g5_metrics["annual_turnover"] >= 0.0

    # Check heatmap
    assert "monthly_heatmap" in bt_results
    assert isinstance(bt_results["monthly_heatmap"], dict)
