"""
Quant Factor Research & Interactive Backtesting Dashboard
Institutional-grade, zero-lookahead Streamlit console featuring:
- Tab 1: Factor Quality Diagnostics (Rank IC, Cumulative IC, Decay)
- Tab 2: Stratified Quantile Backtest (NAV Curves, Long-Short Heatmap, Turnover)
- Tab 3: Universe & Portfolio Holdings (Top/Bottom Baskets, Industry Exposures)
- One-click HTML Report Exporter
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from engine.analytics.evaluator import calculate_rank_ic, run_quantile_backtest
from engine.db import get_connection, get_earliest_trade_date, get_latest_trade_date
from engine.etl.collector import auto_sync_on_startup, sync_market_data
from engine.pit_query import get_tradable_mask

# Set page layout configuration
st.set_page_config(
    page_title="A-Share Quant Factor Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-title { font-size: 0.85rem; color: #6c757d; font-weight: bold; }
    .metric-value { font-size: 1.4rem; color: #212529; font-weight: bold; margin-top: 4px; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; padding: 8px 16px; }
</style>
""", unsafe_allow_html=True)


def get_db_factor_state() -> Tuple[int, str]:
    """Returns (count, max_trade_date) to serve as a reactive cache key."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), MAX(trade_date) FROM factor_metrics;")
    row = cursor.fetchone()
    conn.close()
    return (row[0] or 0, row[1] or "")


@st.cache_data(ttl=3600)
def _load_available_factors_and_dates_cached(db_state: Tuple[int, str]) -> Tuple[List[str], List[str]]:
    """Loads distinct factor names and trading dates from SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT factor_name FROM factor_metrics ORDER BY factor_name;")
    factors = [r[0] for r in cursor.fetchall()]
    cursor.execute("SELECT DISTINCT trade_date FROM factor_metrics ORDER BY trade_date ASC;")
    dates = [r[0] for r in cursor.fetchall()]
    conn.close()
    if not factors:
        factors = ["MOM_20D", "REV_5D", "VOL_20D", "EP_TTM", "ROE_TTM"]
    return factors, dates


def load_available_factors_and_dates() -> Tuple[List[str], List[str]]:
    state = get_db_factor_state()
    return _load_available_factors_and_dates_cached(state)


@st.cache_data(ttl=3600)
def get_cached_rank_ic(factor_name: str, start_date: str, end_date: str, use_clean: bool):
    return calculate_rank_ic(
        factor_name=factor_name,
        lags=[1, 5, 20],
        start_date=start_date,
        end_date=end_date,
        use_clean_val=use_clean,
    )


@st.cache_data(ttl=3600)
def get_cached_backtest(
    factor_name: str,
    start_date: str,
    end_date: str,
    n_groups: int,
    rebalance_interval: int,
    fee_rate: float,
    use_clean: bool,
):
    return run_quantile_backtest(
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        n_groups=n_groups,
        rebalance_interval=rebalance_interval,
        fee_rate=fee_rate,
        use_clean_val=use_clean,
    )


@st.cache_data(ttl=3600)
def get_latest_holdings_data(trade_date: str, factor_name: str, n_groups: int, use_clean: bool):
    conn = get_connection()
    val_col = "clean_val" if use_clean else "raw_val"
    sql = f"""
        SELECT f.ts_code, b.name, b.industry, b.market_board, f.{val_col} as factor_val
        FROM factor_metrics f
        JOIN stock_basic b ON f.ts_code = b.ts_code
        WHERE f.factor_name = ? AND f.trade_date = ?
        ORDER BY f.{val_col} ASC;
    """
    cursor = conn.cursor()
    cursor.execute(sql, (factor_name, trade_date))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    mask_df = get_tradable_mask(trade_date)
    mask_map = {r["ts_code"]: r for r in mask_df.to_dicts()}

    for r in rows:
        m = mask_map.get(r["ts_code"], {})
        r["is_tradable"] = m.get("is_tradable", True)
        r["filter_reason"] = m.get("filter_reason", "OK")

    return rows


# ==========================================
# Startup Lifecycle: Incremental Sync for 5 Years
# ==========================================
if "has_synced" not in st.session_state:
    with st.spinner("🚀 后台启动中：正在自动检测并同步最新未获取的近 5 年行情与财报数据..."):
        sync_res = auto_sync_on_startup(years=5)
        st.session_state["has_synced"] = True
        st.session_state["sync_info"] = sync_res

# ==========================================
# Sidebar Controls
# ==========================================
st.sidebar.title("⚙️ 因子与回测配置")

# 1. Data Warehouse Status & Incremental Update Button
latest_trade_d = get_latest_trade_date() or "暂无数据"
earliest_trade_d = get_earliest_trade_date() or "暂无数据"
with st.sidebar.expander("📦 数据仓储与增量同步状态", expanded=False):
    st.markdown(f"**覆盖范围**: {earliest_trade_d} ~ {latest_trade_d} (近5年)")
    if st.button("🔄 立即拉取最新未同步数据", use_container_width=True):
        with st.spinner("正在增量拉取最新数据并计算因子..."):
            sync_res = sync_market_data(years=5)
            st.cache_data.clear()
            st.success(f"同步完成！新增数据: {sync_res.get('bars_added', 0)} 根 K 线")
            st.rerun()

factors, all_dates = load_available_factors_and_dates()
if not all_dates:
    st.error("数据库中未检测到因子数据，请先同步数据。")
    st.stop()

selected_factor = st.sidebar.selectbox("选择目标因子", factors, index=0)
value_mode = st.sidebar.radio("因子处理模式", ["中性化残差 (clean_val)", "原始值 (raw_val)"], index=0)
use_clean_val = (value_mode == "中性化残差 (clean_val)")

# Date Range Picker
min_d = datetime.strptime(all_dates[0], "%Y%m%d").date()
max_d = datetime.strptime(all_dates[-1], "%Y%m%d").date()
start_d = st.sidebar.date_input("回测起始日期", min_d, min_value=min_d, max_value=max_d)
end_d = st.sidebar.date_input("回测终止日期", max_d, min_value=min_d, max_value=max_d)

start_date_str = start_d.strftime("%Y%m%d")
end_date_str = end_d.strftime("%Y%m%d")

# Backtest parameters
n_groups = st.sidebar.slider("分组数量 (Quantile Groups)", min_value=2, max_value=10, value=10, step=1)
cycle_option = st.sidebar.selectbox("调仓周期 (Rebalance Cycle)", ["日度调仓 (1D)", "周度调仓 (5D)", "月度调仓 (20D)"], index=1)
interval_map = {"日度调仓 (1D)": 1, "周度调仓 (5D)": 5, "月度调仓 (20D)": 20}
rebal_interval = interval_map[cycle_option]

fee_pct = st.sidebar.number_input("单边交易摩擦成本 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
fee_rate = fee_pct / 100.0

st.sidebar.markdown("---")
st.sidebar.caption("🛡️ **防未来函数声明**: 收益严格对齐 $T+1 \\to T+2$ 后复权价，一字涨跌停交易限制与停牌顺延生效。")

# ==========================================
# Main Header
# ==========================================
st.title(f"📊 A股量化因子投研平台: {selected_factor}")
st.caption(f"评估周期: {start_date_str} ~ {end_date_str} | 分组数: {n_groups} | 调仓: {cycle_option} | 摩擦: {fee_pct}%")

tab1, tab2, tab3 = st.tabs(["📈 Tab 1: 因子质量诊断", "🚀 Tab 2: 分层回测表现", "🏢 Tab 3: 股票池与持仓"])

# ==========================================
# Tab 1: Factor Diagnostics
# ==========================================
with tab1:
    st.subheader("1. 因子 Rank IC 检验与衰减分析")
    ic_data = get_cached_rank_ic(selected_factor, start_date_str, end_date_str, use_clean_val)

    if not ic_data or 1 not in ic_data["summary"]:
        st.warning("所选日期范围内可计算样本不足。")
    else:
        s1 = ic_data["summary"].get(1, {})
        s5 = ic_data["summary"].get(5, {})
        s20 = ic_data["summary"].get(20, {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rank IC 均值 (Lag 1)", f"{s1.get('ic_mean', 0.0):.4f}")
        with c2:
            st.metric("年化 IC-IR (Lag 1)", f"{s1.get('ic_ir', 0.0):.2f}")
        with c3:
            st.metric("IC 胜率 (IC > 0)", f"{s1.get('positive_ratio', 0.0):.1f}%")
        with c4:
            st.metric("有效截面期数", f"{s1.get('n_samples', 0)}")

        # Row 1: IC Series & Cumulative IC
        col_left, col_right = st.columns(2)
        with col_left:
            dates_lag1 = ic_data["dates"][1]
            ic_series_lag1 = ic_data["ic_series"][1]
            fig_ic = go.Figure()
            fig_ic.add_trace(go.Bar(
                x=dates_lag1, y=ic_series_lag1,
                marker_color=["#2ecc71" if y > 0 else "#e74c3c" for y in ic_series_lag1],
                name="Rank IC"
            ))
            fig_ic.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_ic.update_layout(title="Rank IC 时序柱状图 (Lag 1)", xaxis_title="交易日期", yaxis_title="Spearman IC")
            st.plotly_chart(fig_ic, use_container_width=True)

        with col_right:
            cum_ic = s1.get("cum_ic", [])
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(x=dates_lag1, y=cum_ic, mode="lines", line=dict(color="#2980b9", width=2), name="累积 IC"))
            fig_cum.update_layout(title="累积 Rank IC 曲线", xaxis_title="交易日期", yaxis_title="Cumulative IC")
            st.plotly_chart(fig_cum, use_container_width=True)

        # Row 2: IC Decay
        lags = [1, 5, 20]
        decay_means = [ic_data["summary"].get(lg, {}).get("ic_mean", 0.0) for lg in lags]
        decay_irs = [ic_data["summary"].get(lg, {}).get("ic_ir", 0.0) for lg in lags]

        fig_decay = go.Figure()
        fig_decay.add_trace(go.Bar(x=[f"Lag {lg}D" for lg in lags], y=decay_means, name="IC 均值", marker_color="#34495e"))
        fig_decay.update_layout(title="因子预测能力时序衰减 (IC Decay)", yaxis_title="IC 均值")
        st.plotly_chart(fig_decay, use_container_width=True)

# ==========================================
# Tab 2: Stratified Backtest Performance
# ==========================================
with tab2:
    st.subheader(f"2. {n_groups} 分组分层回测与多空对冲绩效")
    bt_data = get_cached_backtest(
        selected_factor, start_date_str, end_date_str, n_groups, rebal_interval, fee_rate, use_clean_val
    )

    if not bt_data or "nav_history" not in bt_data:
        st.warning("回测数据不足，请调整日期区间或增加调仓样本。")
    else:
        dates = bt_data["dates"]
        nav_hist = bt_data["nav_history"]
        ls_nav = bt_data["long_short_nav"]
        bm_nav = bt_data["benchmark_nav"]
        metrics = bt_data["metrics"]

        # 1. Cumulative NAV Curves
        fig_nav = go.Figure()
        colors = px.colors.sample_colorscale("Viridis", [i / (n_groups - 1) for i in range(n_groups)])
        for g in range(1, n_groups + 1):
            curve = nav_hist[g][:len(dates)]
            fig_nav.add_trace(go.Scatter(
                x=dates, y=curve, mode="lines",
                name=f"Group {g} (Q{g})",
                line=dict(color=colors[g - 1], width=3 if g in (1, n_groups) else 1.5)
            ))
        # Benchmark
        fig_nav.add_trace(go.Scatter(
            x=dates, y=bm_nav[:len(dates)], mode="lines",
            name="基准 (等权)", line=dict(color="gray", dash="dash", width=1.5)
        ))
        fig_nav.update_layout(
            title=f"分层累计净值走势 ({n_groups} 分组, 扣费 {fee_pct}%)",
            xaxis_title="调仓结算日", yaxis_title="净值 (NAV)"
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        # 2. Long-Short Spread Curve & Monotonicity
        c_ls1, c_ls2 = st.columns([3, 1])
        with c_ls1:
            fig_ls = go.Figure()
            fig_ls.add_trace(go.Scatter(
                x=dates, y=ls_nav[:len(dates)], mode="lines",
                name="多空对冲 (Top - Bottom)", line=dict(color="#e67e22", width=2.5)
            ))
            fig_ls.update_layout(title="多空对冲净值走势 (Q_Top / Q_Bottom)", xaxis_title="日期", yaxis_title="多空累计倍数")
            st.plotly_chart(fig_ls, use_container_width=True)

        with c_ls2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            mono_score = bt_data.get("monotonicity_score", 0.0)
            st.markdown(f"<div class='metric-title'>分层单调性评分</div><div class='metric-value'>{mono_score:+.2f}</div>", unsafe_allow_html=True)
            st.caption("分组序号与最终收益的秩相关系数。越接近 +1.0 表示单调正向区分度越强。")
            st.markdown("</div>", unsafe_allow_html=True)

            ls_m = metrics.get("Long_Short", {})
            st.markdown("<div class='metric-card' style='margin-top:12px;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-title'>多空年化收益率</div><div class='metric-value'>{ls_m.get('annual_return', 0.0):+.2f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-title'>多空夏普比率</div><div class='metric-value'>{ls_m.get('sharpe_ratio', 0.0):.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-title'>多空最大回撤</div><div class='metric-value'>{ls_m.get('max_drawdown', 0.0):.2f}%</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Performance Metrics Table
        st.write("#### 🏆 各分层及多空组合核心指标看板")
        table_rows = []
        for g in range(1, n_groups + 1):
            m = metrics.get(f"Group_{g}", {})
            table_rows.append({
                "组合分层": f"Group {g} (Q{g})",
                "年化收益率 (%)": f"{m.get('annual_return', 0.0):+.2f}%",
                "年化波动率 (%)": f"{m.get('annual_volatility', 0.0):.2f}%",
                "夏普比率": f"{m.get('sharpe_ratio', 0.0):.2f}",
                "最大回撤 (%)": f"{m.get('max_drawdown', 0.0):.2f}%",
                "卡玛比率": f"{m.get('calmar_ratio', 0.0):.2f}",
                "年化换手率 (X)": f"{m.get('annual_turnover', 0.0):.1f}x",
                "期末净值": f"{m.get('final_nav', 1.0):.4f}",
            })
        # Add Long-Short row
        table_rows.append({
            "组合分层": "多空对冲 (Top-Bottom)",
            "年化收益率 (%)": f"{ls_m.get('annual_return', 0.0):+.2f}%",
            "年化波动率 (%)": f"{ls_m.get('annual_volatility', 0.0):.2f}%",
            "夏普比率": f"{ls_m.get('sharpe_ratio', 0.0):.2f}",
            "最大回撤 (%)": f"{ls_m.get('max_drawdown', 0.0):.2f}%",
            "卡玛比率": f"{ls_m.get('calmar_ratio', 0.0):.2f}",
            "年化换手率 (X)": "-",
            "期末净值": f"{ls_m.get('final_nav', 1.0):.4f}",
        })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        # 4. Monthly Heatmap for Long-Short
        st.write("#### 📅 多空对冲月度收益率热力图 (%)")
        heatmap_dict = bt_data.get("monthly_heatmap", {})
        if heatmap_dict:
            months = [f"{m:02d}" for m in range(1, 13)]
            years = sorted(heatmap_dict.keys())
            grid_data = []
            for y in years:
                row = [heatmap_dict[y].get(m, np.nan) for m in months]
                grid_data.append(row)

            fig_hm = go.Figure(data=go.Heatmap(
                z=grid_data, x=[f"{int(m)}月" for m in months], y=years,
                colorscale="RdYlGn", colorbar=dict(title="%"), text=grid_data, texttemplate="%{text:.1f}%"
            ))
            fig_hm.update_layout(title="多空组合月度收益矩阵", yaxis_autorange="reversed")
            st.plotly_chart(fig_hm, use_container_width=True)

# ==========================================
# Tab 3: Universe & Portfolio Holdings
# ==========================================
with tab3:
    st.subheader("3. 最新截面持仓明细与行业分布")
    latest_date = all_dates[-1]
    st.write(f"当前截面日期: **{latest_date}**")

    holdings_rows = get_latest_holdings_data(latest_date, selected_factor, n_groups, use_clean_val)
    if not holdings_rows:
        st.info("当前日期无截面持仓数据。")
    else:
        df_holdings = pd.DataFrame(holdings_rows)

        # Top Group (Long) and Bottom Group (Short)
        group_sz = len(df_holdings) // n_groups
        df_bottom = df_holdings.iloc[:group_sz]
        df_top = df_holdings.iloc[-group_sz:]

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.write(f"🟢 **多头候选组合 (Top Quantile, {len(df_top)} 标的)**")
            st.dataframe(
                df_top[["ts_code", "name", "industry", "factor_val", "filter_reason"]].rename(
                    columns={"ts_code": "代码", "name": "名称", "industry": "行业", "factor_val": "因子值", "filter_reason": "过滤标记"}
                ),
                use_container_width=True
            )

        with col_h2:
            st.write(f"🔴 **空头候选组合 (Bottom Quantile, {len(df_bottom)} 标的)**")
            st.dataframe(
                df_bottom[["ts_code", "name", "industry", "factor_val", "filter_reason"]].rename(
                    columns={"ts_code": "代码", "name": "名称", "industry": "行业", "factor_val": "因子值", "filter_reason": "过滤标记"}
                ),
                use_container_width=True
            )

        # Industry Exposure breakdown
        st.write("#### 🏭 申万一级行业配置分布 (Top 组合)")
        ind_counts = df_top["industry"].value_counts().reset_index()
        ind_counts.columns = ["行业", "个股数量"]
        fig_ind = px.bar(ind_counts, x="行业", y="个股数量", color="个股数量", color_continuous_scale="Blues")
        st.plotly_chart(fig_ind, use_container_width=True)

# ==========================================
# Report Export Feature
# ==========================================
st.markdown("---")
st.subheader("📑 因子研究简报一键导出")


def generate_html_report(factor_name: str, start: str, end: str, metrics: Dict) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ls_info = metrics.get("Long_Short", {})
    q10_info = metrics.get("Group_10", metrics.get("Group_5", {}))
    q1_info = metrics.get("Group_1", {})

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>A-Share Quant Factor Report: {factor_name}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; color: #222; background: #fafafa; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
            .tag {{ display: inline-block; background: #edf2f7; padding: 4px 10px; border-radius: 4px; font-size: 0.9em; margin-right: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #cbd5e0; padding: 10px 14px; text-align: left; }}
            th {{ background: #f7fafc; }}
            .highlight {{ color: #2b6cb0; font-weight: bold; }}
            .footer {{ margin-top: 30px; font-size: 0.85em; color: #718096; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>A-Share 因子投研诊断简报: {factor_name}</h1>
            <div>
                <span class="tag">回测区间: {start} ~ {end}</span>
                <span class="tag">生成时间: {now_str}</span>
                <span class="tag">框架: Python + SQLite + Polars</span>
            </div>
            <h3>核心对冲与分层绩效概览</h3>
            <table>
                <tr>
                    <th>指标名称</th>
                    <th>多空对冲 (Top - Bottom)</th>
                    <th>多头组 (Top Quantile)</th>
                    <th>空头组 (Bottom Quantile)</th>
                </tr>
                <tr>
                    <td>年化收益率</td>
                    <td class="highlight">{ls_info.get('annual_return', 0.0):+.2f}%</td>
                    <td>{q10_info.get('annual_return', 0.0):+.2f}%</td>
                    <td>{q1_info.get('annual_return', 0.0):+.2f}%</td>
                </tr>
                <tr>
                    <td>夏普比率 (Sharpe)</td>
                    <td class="highlight">{ls_info.get('sharpe_ratio', 0.0):.2f}</td>
                    <td>{q10_info.get('sharpe_ratio', 0.0):.2f}</td>
                    <td>{q1_info.get('sharpe_ratio', 0.0):.2f}</td>
                </tr>
                <tr>
                    <td>最大回撤 (MDD)</td>
                    <td>{ls_info.get('max_drawdown', 0.0):.2f}%</td>
                    <td>{q10_info.get('max_drawdown', 0.0):.2f}%</td>
                    <td>{q1_info.get('max_drawdown', 0.0):.2f}%</td>
                </tr>
                <tr>
                    <td>卡玛比率 (Calmar)</td>
                    <td>{ls_info.get('calmar_ratio', 0.0):.2f}</td>
                    <td>{q10_info.get('calmar_ratio', 0.0):.2f}</td>
                    <td>{q1_info.get('calmar_ratio', 0.0):.2f}</td>
                </tr>
            </table>
            <div class="footer">
                <p>※ 严格防止未来函数：收益率对齐 T+1 至 T+2 开盘后复权价，一字涨跌停交易限制与停牌顺延逻辑生效。</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html


if "metrics" in locals() and metrics:
    report_html = generate_html_report(selected_factor, start_date_str, end_date_str, metrics)
    b64 = base64.b64encode(report_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="factor_report_{selected_factor}.html" style="text-decoration:none;"><button style="background-color:#2980b9;color:white;padding:10px 20px;border:none;border-radius:4px;cursor:pointer;font-weight:bold;">📥 导出 HTML 研报</button></a>'
    st.markdown(href, unsafe_allow_html=True)
