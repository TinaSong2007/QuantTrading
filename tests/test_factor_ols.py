"""
Tests for Factor Processing: MAD Winzorization, Z-Score,
Vectorized OLS Neutralization Orthogonality, NaN Isolation, and <50ms Speed Benchmark.
"""

import time
import numpy as np
import polars as pl
import pytest

from engine.factors.processor import (
    SW_INDUSTRIES_31,
    clean_factor_cross_section,
    neutralize_ols,
    standardize_zscore,
    winsorize_mad,
)


def test_mad_winzorization():
    # Array with extreme outlier
    arr = np.array([1.0, 1.1, 0.9, 1.0, 1.2, 0.95, 100.0, -100.0])
    winz = winsorize_mad(arr, n=3.0)
    assert winz.max() < 10.0
    assert winz.min() > -10.0


def test_ols_neutralization_orthogonality():
    np.random.seed(42)
    n = 1000
    y = np.random.normal(0, 1, n)
    industries = np.random.choice(SW_INDUSTRIES_31, n)
    mv = np.random.uniform(1e9, 1e11, n)

    # Induce strong correlation with industry and market cap
    ind_effect = np.array([hash(ind) % 5 for ind in industries], dtype=float)
    cap_effect = np.log(mv) * 0.5
    y_correlated = y + ind_effect + cap_effect

    residuals, X = neutralize_ols(y_correlated, industries, mv)

    # Assert X has exactly 32 columns (31 industries + 1 log_mv), NO duplicate intercept
    assert X.shape == (n, 32)

    # Verify orthogonality: Pearson correlation with each column of X must be < 1e-5
    res_std = np.std(residuals)
    assert res_std > 1e-6

    for j in range(X.shape[1]):
        col = X[:, j]
        col_std = np.std(col)
        if col_std > 1e-8:
            corr = np.corrcoef(residuals, col)[0, 1]
            assert abs(corr) < 1e-5, f"Column {j} correlation with residual is {corr:.2e} >= 1e-5"


def test_nan_isolation():
    # DataFrame with missing values
    df = pl.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ"],
        "raw_val": [1.5, None, 2.0, 3.5, 1.8, 2.2],
        "industry": ["银行", "房地产", None, "医药生物", "电子", "汽车"],
        "total_mv": [1e10, 2e10, 3e10, None, 5e10, 6e10],
    })

    cleaned = clean_factor_cross_section(df)
    # Only rows with valid raw_val, industry, and total_mv should be processed
    assert cleaned.height == 3
    # Check that output has no NaNs
    assert cleaned["clean_val"].null_count() == 0
    assert not np.isnan(cleaned["clean_val"].to_numpy()).any()


def test_ols_speed_benchmark_5000_stocks():
    """
    Asserts that a full cross-section of 5,000 stocks cleans & neutralizes in < 50ms.
    """
    np.random.seed(123)
    n = 5000
    df = pl.DataFrame({
        "ts_code": [f"{i:06d}.SH" for i in range(n)],
        "raw_val": np.random.normal(0.05, 0.2, n),
        "industry": np.random.choice(SW_INDUSTRIES_31, n),
        "total_mv": np.random.uniform(5e8, 5e11, n),
    })

    start_time = time.perf_counter()
    cleaned = clean_factor_cross_section(df)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[Benchmark] 5,000 stocks cross-sectional OLS completed in: {elapsed_ms:.2f} ms")
    assert cleaned.height == 5000
    # Strict speed assertion: must be < 50ms
    assert elapsed_ms < 50.0, f"Execution took {elapsed_ms:.2f}ms, which exceeds 50ms target"
