# A-Share Quant Factor Platform (Python + SQLite + Polars + Streamlit)

> 基于 **Python 3.11+ / SQLite (WAL) / Polars / NumPy / Streamlit** 的机构级 A 股量化因子投研与分层回测框架。严格恪守无未来函数（Anti-Lookahead Bias）契约，内嵌涨跌停交易限制、非交易日顺延（Carry-Over）及 OLS 纯净正交中性化。

---

## 🌟 核心量化设计与避坑特性

1. **零配置单文件数据库（`warehouse.db`）**：
   - 开启 `PRAGMA journal_mode=WAL;`、`PRAGMA synchronous=NORMAL;`、`PRAGMA cache_size=-64000;`（64MB 缓存）与 `PRAGMA temp_store=MEMORY;`。
   - 连接统一配置 `timeout=30.0`，彻底规避高并发与 Streamlit 刷新时的 `database is locked` 报错。
2. **点对点 (PIT) 财报披露与防穿越（Anti-Lookahead）**：
   - 严格限定 `ann_date <= trade_date`；同日若存在多次披露按 `end_date DESC` 取最新期。
   - ETL 阶段预计算滚动单季化与 TTM 指标（`ttm_net_profit`, `roe_ttm`），避免动态运行时自连接性能瓶颈。
3. **后复权收益率计算（Data Clean）**：
   - 维护累计复权因子 `adj_factor` 与后复权价 `adj_close = close * adj_factor`，严格使用后复权序列计算收益率，杜绝除权除息跳空造成的虚假极值。
4. **动态交易掩码（Trading Mask）**：
   - 自动过滤 ST/\*ST、上市不足 120 天新股、当日停牌（`vol == 0`）。
   - **一字涨跌停交易限制**：买入日 $T+1$ 开盘若 `open == high` 且开盘涨幅 $\ge 9.8\%$（主板）或 $19.8\%$（双创）判定为一字涨停无法买入；卖出日一字跌停无法卖出。
5. **停牌股顺延与权重再平衡（Carry-Over）**：
   - 调仓日若持仓股票停牌或跌停无法卖出，原有仓位被动顺延保留（Carry-Over），可用资金仅在其余可交易标的间重新归一化分配。
6. **无截距 OLS 纯矩阵正交中性化（Linear Algebra）**：
   - 使用 `np.linalg.lstsq` 进行截面投影，$X = [31 \text{ 申万行业哑变量}, \ln(\text{总市值})]$，**不添加全 1 截距常数列**，避免多重共线性。
   - 前置 Polars 严格隔离 NaN 缺失值，5,000 标的截面计算仅需 **7.16ms**（远超 <50ms 规范要求），残差与自变量皮尔逊相关系数收敛至 $< 10^{-5}$。
7. **完整买方因子分析套件（Analytics）**：
   - Rank IC（Spearman）、累积 IC、年化 IC-IR（$\frac{\text{Mean}}{\text{Std}} \times \sqrt{252}$）与多周期 IC 衰减。
   - 10 分组分层回测，扣除单边 0.1% 交易摩擦成本。
   - 单期与年化换手率（Turnover）监控、多空组合（Top 10% - Bottom 10%）月度收益率热力图。
8. **近 5 年历史全覆盖与启动自动增量同步（Incremental Sync）**：
   - 支持跨度达近 5 年（2021 ~ 2026）的日频行情与季度 PIT 财报仓储；
   - 每次后台/Streamlit 启动时，自动查询现有数据库最大交易日 $T_{max}$，仅拉取增量缺失交易日并保持价格状态连续性；
   - 支持 CLI 断点续传指令：`python -m engine.etl.collector --sync --years 5`，无重复拉取。
9. **Streamlit 响应式交互看板（秒级交互）**：
   - 核心函数全量包裹 `@st.cache_data`，实现参数拖拽时的秒级无感渲染；
   - 侧边栏实时显示仓储最新覆盖日期与「一键增量同步最新数据」按钮；
   - 支持 HTML 因子投研简报一键导出。

---

## 📁 目录架构

```
QuantTrading/
├── .cursorrules                  # 量化编码与防翻车规范（包含 4 条量化实盘硬约束）
├── requirements.txt              # 核心依赖 (Polars, NumPy, SciPy, Streamlit, Plotly, AkShare, Pytest)
├── warehouse.db                  # SQLite 仓储数据库 (WAL 高性能模式)
├── app.py                        # Streamlit 3-Tab 交互式因子投研看板与报告导出
├── engine/
│   ├── db.py                     # SQLite 连接管理器、WAL PRAGMA 及 4 张契约表 Schema
│   ├── pit_query.py              # PIT 财报时序合并与动态交易掩码 (ST/新股/停牌/一字涨跌停)
│   ├── etl/
│   │   ├── collector.py          # ETL 管道 (分批 5000 行 executemany, 离线/在线双轨)
│   │   └── generator.py          # 高保真行情与财报生成器 (含后复权、分红、预计算 TTM)
│   ├── factors/
│   │   ├── processor.py          # 纯向量化 MAD 去极值、Z-Score、OLS 行业市值中性化
│   │   └── definitions.py        # 常用因子库 (MOM_20D, REV_5D, VOL_20D, EP_TTM, ROE_TTM)
│   └── analytics/
│       └── evaluator.py          # Rank IC (Lag 1/5/20)、10 分组回测、摩擦成本与换手率统计
└── tests/
    ├── test_db_wal.py            # SQLite WAL 模式与复合主键约束测试
    ├── test_pit_lookahead.py     # 严格防未来函数与交易掩码测试
    ├── test_factor_ols.py        # OLS 残差正交性 (<1e-5)、NaN 隔离与 <50ms 性能基准测试
    └── test_evaluator.py         # 收益率时序对齐、换手率与 Carry-Over 逻辑测试
```

---

## 🚀 快速启动指南

### 1. 激活虚拟环境与安装依赖

```bash
# 激活当前虚拟环境
source .venv/bin/activate

# 或全新安装依赖
pip install -r requirements.txt
```

### 2. 执行自动化全量测试套件 (TDD)

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v -s
```

输出示例：
```text
tests/test_db_wal.py::test_wal_pragmas_and_schema PASSED
tests/test_db_wal.py::test_composite_primary_key_enforcement PASSED
tests/test_evaluator.py::test_rank_ic_computation PASSED
tests/test_evaluator.py::test_quantile_backtest_with_friction_and_turnover PASSED
tests/test_factor_ols.py::test_mad_winzorization PASSED
tests/test_factor_ols.py::test_ols_neutralization_orthogonality PASSED
tests/test_factor_ols.py::test_nan_isolation PASSED
tests/test_factor_ols.py::test_ols_speed_benchmark_5000_stocks 
[Benchmark] 5,000 stocks cross-sectional OLS completed in: 7.16 ms PASSED
tests/test_pit_lookahead.py::test_anti_lookahead_financial_pit PASSED
tests/test_pit_lookahead.py::test_dynamic_tradable_mask_and_limit_up PASSED

============================== 10 passed in 3.00s ==============================
```

### 3. 数据更新与因子计算 (ETL CLI)

```bash
# 查看数据库各表当前行数
PYTHONPATH=. .venv/bin/python -m engine.etl.collector --status

# 生成/重置 100 只标的 3 年历史行情与 PIT 财报
PYTHONPATH=. .venv/bin/python -m engine.etl.collector --sample --stocks 100

# 在线拉取真实沪深 300 数据 (需外网环境)
PYTHONPATH=. .venv/bin/python -m engine.etl.collector --online --stocks 30
```

### 4. 启动 Streamlit 可视化投研控制台

```bash
.venv/bin/streamlit run app.py
```

在浏览器打开终端打印的地址（默认 `http://localhost:8501`），体验：
- **Tab 1: 因子质量诊断**（Rank IC 时序、累积 IC、多周期 IC 衰减）
- **Tab 2: 分层回测表现**（10 分组净值曲线、多空对冲走势、换手率监控、月度收益热力图）
- **Tab 3: 股票池与持仓**（最新调仓日 Top/Bottom 个股明细与申万行业配置占比）
- **导出研报**：点击「📥 导出 HTML 研报」一键生成独立报告文件。
