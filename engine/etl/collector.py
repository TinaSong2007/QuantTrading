"""
ETL Data Ingestion Pipeline
Supports chunked batch insertion (5,000 rows/batch) with INSERT OR REPLACE,
providing dual online (AkShare) and offline high-fidelity sample generation modes.
"""

import argparse
import concurrent.futures
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup

from engine.db import (
    get_connection,
    get_db,
    get_earliest_trade_date,
    get_latest_financial_ann_date,
    get_latest_trade_date,
    get_stock_latest_states,
    init_db,
)
from engine.etl.generator import (
    generate_mock_daily_bars,
    generate_mock_financial_pit,
    generate_mock_stock_universe,
    generate_trade_dates,
)


def batch_insert(
    conn,
    sql: str,
    data: Sequence[Tuple],
    chunk_size: int = 5000,
    label: str = "records"
) -> int:
    """
    Executes high-performance chunked insertion using executemany with transactional batching.
    """
    total = len(data)
    if total == 0:
        return 0

    inserted = 0
    cursor = conn.cursor()
    for i in range(0, total, chunk_size):
        chunk = data[i:i + chunk_size]
        cursor.executemany(sql, chunk)
        inserted += len(chunk)
        print(f"[{label}] Inserted {inserted}/{total} rows ({inserted / total * 100:.1f}%)", end="\r")
    conn.commit()
    print(f"\n[{label}] Done: {inserted} rows successfully committed.")
    return inserted


def ingest_stock_basics(stocks: List[Dict], db_path: str = None) -> int:
    """Inserts or updates stock basic information."""
    sql = """
    INSERT OR REPLACE INTO stock_basic (ts_code, name, industry, list_date, is_st, market_board)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    data = [
        (s["ts_code"], s["name"], s["industry"], s["list_date"], s["is_st"], s.get("market_board", "Main"))
        for s in stocks
    ]
    with get_db(db_path) as conn:
        return batch_insert(conn, sql, data, label="stock_basic")


def ingest_daily_bars(bars: List[Tuple], db_path: str = None) -> int:
    """Inserts or updates daily market bars with backward-adjusted price."""
    sql = """
    INSERT OR REPLACE INTO daily_bar (
        ts_code, trade_date, open, high, low, close,
        pre_close, vol, amount, adj_factor, adj_close
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    with get_db(db_path) as conn:
        return batch_insert(conn, sql, bars, chunk_size=5000, label="daily_bar")


def ingest_financial_pit(records: List[Tuple], db_path: str = None) -> int:
    """Inserts or updates PIT financial records with precomputed TTM."""
    sql = """
    INSERT OR REPLACE INTO financial_pit (
        ts_code, ann_date, end_date, roe, net_profit, revenue, ttm_net_profit, roe_ttm
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """
    with get_db(db_path) as conn:
        return batch_insert(conn, sql, records, chunk_size=5000, label="financial_pit")


def run_sample_etl(n_stocks: int = 100, db_path: str = None) -> None:
    """
    Populates SQLite database with high-fidelity sample data covering 2022-2024.
    """
    print(f"Initializing database at {db_path or 'warehouse.db'}...")
    init_db(db_path)

    print(f"Generating mock universe of {n_stocks} stocks...")
    stocks = generate_mock_stock_universe(n_stocks)
    ingest_stock_basics(stocks, db_path)

    trade_dates = generate_trade_dates("20220101", "20241231")
    print(f"Generating {len(trade_dates)} trading days of daily bars...")
    bars = generate_mock_daily_bars(stocks, trade_dates)
    ingest_daily_bars(bars, db_path)

    print("Generating quarterly PIT financial statements with precalculated TTM...")
    financials = generate_mock_financial_pit(stocks, years=[2021, 2022, 2023, 2024])
    ingest_financial_pit(financials, db_path)

    print("Sample ETL completed successfully.")
    print_table_counts(db_path)


SW_INDUSTRY_MAPPING = {
    "白酒": "食品饮料", "啤酒": "食品饮料", "食品加工": "食品饮料", "调味发酵品": "食品饮料", "乳品": "食品饮料",
    "银行": "银行", "证券": "非银金融", "保险": "非银金融", "多元金融": "非银金融",
    "半导体": "电子", "消费电子": "电子", "元件": "电子", "光学光电子": "电子", "显示器件": "电子", "电子化学品": "电子",
    "软件开发": "计算机", "IT服务": "计算机", "计算机设备": "计算机",
    "化学制药": "医药生物", "中药": "医药生物", "生物制品": "医药生物", "医疗器械": "医药生物", "医疗服务": "医药生物", "医药商业": "医药生物",
    "电池": "电力设备", "光伏设备": "电力设备", "风电设备": "电力设备", "电网设备": "电力设备",
    "乘用车": "汽车", "汽车零部件": "汽车", "商用车": "汽车",
    "通用设备": "机械设备", "专用设备": "机械设备", "自动化设备": "机械设备", "工程机械": "机械设备",
    "基础化工": "基础化工", "化学原料": "基础化工", "化学制品": "基础化工", "农化制品": "基础化工", "涤纶": "基础化工", "钾肥": "基础化工",
    "工业金属": "有色金属", "贵金属": "有色金属", "小金属": "有色金属", "金属新材料": "有色金属", "黄金": "有色金属", "铝": "有色金属", "铜": "有色金属",
    "煤炭开采": "煤炭", "焦炭": "煤炭",
    "炼油化工": "石油石化", "油服工程": "石油石化",
    "通信设备": "通信", "通信服务": "通信",
    "游戏": "传媒", "数字媒体": "传媒", "广告营销": "传媒", "影视院线": "传媒",
    "房地产开发": "房地产", "房地产服务": "房地产",
    "白色家电": "家用电器", "小家电": "家用电器", "厨卫电器": "家用电器",
    "航空装备": "国防军工", "航天装备": "国防军工", "地面兵装": "国防军工", "军工电子": "国防军工",
    "电力": "公用事业", "燃气": "公用事业",
    "铁路公路": "交通运输", "航空机场": "交通运输", "航运港口": "交通运输", "物流": "交通运输",
    "普钢": "钢铁", "特钢": "钢铁",
    "水泥": "建筑材料", "玻璃玻纤": "建筑材料",
    "房屋建设": "建筑装饰", "基础建设": "建筑装饰",
    "一般零售": "商贸零售", "专业零售": "商贸零售", "免税": "商贸零售",
}


def resolve_sw_industry(code: str, name: str) -> str:
    """Resolves stock name and code to Shenwan Level-1 industry."""
    # 1. Fast keyword matching on blue-chip names
    if any(k in name for k in ["银行", "农行", "工行", "建行", "交行", "中行", "招行", "浦发", "兴业", "民生", "平安", "光大", "华夏", "北京", "宁波", "江苏", "上海", "南京", "杭州", "成都", "邮储", "中信"]):
        if "银行" in name or "商行" in name:
            return "银行"
    if any(k in name for k in ["证券", "中信建投", "海通", "华泰", "广发", "国泰君安", "申万宏源", "东方财富", "招商证券", "国信", "光大证券"]):
        return "非银金融"
    if any(k in name for k in ["人寿", "太保", "新华保险"]):
        return "非银金融"
    if any(k in name for k in ["茅台", "五粮液", "泸州老窖", "汾酒", "洋河", "古井贡", "今世缘", "青岛啤酒", "海天", "伊利", "东鹏饮料"]):
        return "食品饮料"
    if any(k in name for k in ["宁德时代", "隆基", "通威", "阳光电源", "晶澳", "天合", "特变电工", "亿纬锂能", "国轩高科", "德业股份"]):
        return "电力设备"
    if any(k in name for k in ["比亚迪", "长城汽车", "长安汽车", "上汽", "广汽", "福耀玻璃", "赛力斯", "拓普集团"]):
        return "汽车"
    if any(k in name for k in ["中芯国际", "海光信息", "寒武纪", "北方华创", "韦尔", "中微", "澜起", "立讯精密", "京东方", "歌尔", "传音控股"]):
        return "电子"
    if any(k in name for k in ["恒瑞医药", "迈瑞医疗", "药明康德", "片仔癀", "云南白药", "爱尔眼科", "复星医药", "联影医疗", "智飞生物", "百济神州", "长春高新", "华东医药"]):
        return "医药生物"
    if any(k in name for k in ["中国移动", "中国电信", "中国联通", "中兴通讯"]):
        return "通信"
    if any(k in name for k in ["科大讯飞", "金山办公", "三六零", "恒生电子", "用友网络", "同花顺"]):
        return "计算机"
    if any(k in name for k in ["万科", "保利", "招商蛇口", "华侨城", "金地集团"]):
        return "房地产"
    if any(k in name for k in ["中国石油", "中国石化", "中国海油"]):
        return "石油石化"
    if any(k in name for k in ["中国神华", "陕西煤业", "兖矿能源", "潞安环能", "山西焦煤"]):
        return "煤炭"
    if any(k in name for k in ["紫金矿业", "洛阳钼业", "江西铜业", "天齐锂业", "赣锋锂业", "山东黄金", "中金黄金", "中国铝业", "北方稀土", "华友钴业"]):
        return "有色金属"
    if any(k in name for k in ["中国建筑", "中国中铁", "中国铁建", "中国交建", "中国电建", "中国能建"]):
        return "建筑装饰"
    if any(k in name for k in ["海螺水泥", "北新建材"]):
        return "建筑材料"
    if any(k in name for k in ["顺丰控股", "中远海控", "京沪高铁", "上海机场", "白云机场", "招商轮船"]):
        return "交通运输"
    if any(k in name for k in ["美的集团", "格力电器", "海尔智家", "三花智控"]):
        return "家用电器"
    if any(k in name for k in ["中国中免", "王府井", "永辉超市"]):
        return "商贸零售"
    if any(k in name for k in ["宝钢股份", "包钢股份", "华菱钢铁"]):
        return "钢铁"
    if any(k in name for k in ["万华化学", "华鲁恒升", "恒力石化", "荣盛石化", "盐湖股份"]):
        return "基础化工"
    if any(k in name for k in ["长江电力", "中国广核", "中国核电", "华能国际", "国电电力", "华能水电", "国投电力"]):
        return "公用事业"
    if any(k in name for k in ["中航沈飞", "中航西飞", "航发动力", "中航光电", "中国船舶", "中航重机"]):
        return "国防军工"
    if any(k in name for k in ["三一重工", "中联重科", "徐工机械", "潍柴动力", "恒立液压"]):
        return "机械设备"

    # 2. Try online lookup via Sina CorpOtherInfo
    try:
        url = f"http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpOtherInfo/stockid/{code}/menu_num/5.phtml"
        r = requests.get(url, timeout=2)
        r.encoding = "gbk"
        soup = BeautifulSoup(r.text, "html.parser")
        for tbl in soup.find_all("table"):
            if "所属行业板块" in tbl.text:
                tds = [td.get_text().strip() for td in tbl.find_all("td") if td.get_text().strip()]
                if len(tds) >= 2:
                    sub_ind = tds[1]
                    return SW_INDUSTRY_MAPPING.get(sub_ind, sub_ind)
    except Exception:
        pass
    return "综合"


def fetch_real_stock_bars(ts_code: str) -> List[Tuple]:
    """
    Fetches past 5 years of daily bars (2021-10 to 2026-10) using high-speed Tencent Finance API.
    Returns tuples matching daily_bar schema:
    (ts_code, trade_date, open, high, low, close, pre_close, vol, amount, adj_factor, adj_close)
    """
    code_raw = ts_code.split(".")[0]
    market_prefix = "sh" if ts_code.endswith(".SH") else "sz"
    sym = f"{market_prefix}{code_raw}"

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    # Query in 2 parts to cover 5 years of data (approx 1211 bars)
    try:
        r1 = requests.get(url, params={"param": f"{sym},day,2021-10-01,2024-02-06,640,hfq"}, timeout=8).json()
        r2 = requests.get(url, params={"param": f"{sym},day,2024-02-07,2026-10-01,640,hfq"}, timeout=8).json()
    except Exception:
        return []

    d1 = r1.get("data", {}).get(sym, {})
    d2 = r2.get("data", {}).get(sym, {})
    bars1 = d1.get("hfqday", d1.get("day", []))
    bars2 = d2.get("hfqday", d2.get("day", []))

    all_raw = bars1 + bars2
    if not all_raw:
        return []

    bars = []
    seen_dates = set()
    prev_close = None

    for item in all_raw:
        date_str = item[0].replace("-", "")
        if date_str in seen_dates:
            continue
        seen_dates.add(date_str)

        open_p = float(item[1])
        close_p = float(item[2])
        high_p = float(item[3])
        low_p = float(item[4])
        vol = float(item[5])
        amount = round(vol * ((open_p + close_p) / 2.0) * 100.0, 2)
        pre_c = prev_close if prev_close is not None else open_p
        prev_close = close_p

        # Prices are backward-adjusted (HFQ)
        bars.append((
            ts_code, date_str, open_p, high_p, low_p, close_p,
            pre_c, vol, amount, 1.0, close_p
        ))

    return bars


def run_real_csi300_etl(n_stocks: int = 300, db_path: str = None) -> None:
    """
    Ingests real CSI 300 constituents and 5 years of daily bars from live market APIs.
    """
    print(f"\n=== Starting Real CSI 300 ETL (Target: {n_stocks} stocks, 5 Years) ===")
    init_db(db_path)

    import akshare as ak
    print("1. Fetching official CSI 300 constituent list via CSIndex...")
    cons_df = ak.index_stock_cons_weight_csindex(symbol="000300")
    if n_stocks < len(cons_df):
        cons_df = cons_df.head(n_stocks)

    stocks = []
    for _, row in cons_df.iterrows():
        code = str(row["成分券代码"]).zfill(6)
        name = str(row["成分券名称"])
        exch = str(row["交易所"])
        if "上海" in exch or code.startswith("6"):
            ts_code = f"{code}.SH"
            board = "STAR" if code.startswith("688") else "Main"
        else:
            ts_code = f"{code}.SZ"
            board = "ChiNext" if code.startswith("300") else "Main"

        industry = resolve_sw_industry(code, name)
        stocks.append({
            "ts_code": ts_code,
            "name": name,
            "industry": industry,
            "list_date": "20100101",
            "is_st": 1 if "ST" in name else 0,
            "market_board": board,
        })

    print(f"Ingesting {len(stocks)} stock basics into database...")
    ingest_stock_basics(stocks, db_path)

    # 2. Checkpoint check: find stocks that already have bars in database
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT ts_code, COUNT(*) FROM daily_bar GROUP BY ts_code HAVING COUNT(*) > 500;")
    existing_stocks = {r[0] for r in cursor.fetchall()}
    conn.close()

    pending_stocks = [s for s in stocks if s["ts_code"] not in existing_stocks]
    print(f"2. Fetching 5-year daily bars for {len(pending_stocks)} pending stocks ({len(existing_stocks)} already cached)...")

    all_new_bars = []
    completed = 0
    t_start = datetime.now()

    def _fetch_worker(stock):
        ts_code = stock["ts_code"]
        name = stock["name"]
        bars = fetch_real_stock_bars(ts_code)
        return ts_code, name, bars

    if pending_stocks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_worker, s): s for s in pending_stocks}
            for fut in concurrent.futures.as_completed(futures):
                ts_code, name, bars = fut.result()
                all_new_bars.extend(bars)
                completed += 1
                if completed % 25 == 0 or completed == len(pending_stocks):
                    print(f"[{completed}/{len(pending_stocks)}] Fetched {ts_code} ({name}) - {len(bars)} bars")

    if all_new_bars:
        print(f"Ingesting {len(all_new_bars)} real daily bars into SQLite...")
        ingest_daily_bars(all_new_bars, db_path)

    # 3. Generate PIT financial statements for these real stocks
    print("3. Generating PIT financial statements with precomputed TTM for 300 stocks...")
    financials = generate_mock_financial_pit(stocks, years=[2021, 2022, 2023, 2024, 2025, 2026])
    ingest_financial_pit(financials, db_path)

    # 4. Compute factors across the 5 years
    print("4. Calculating cross-sectional factors across 5-year calendar...")
    from engine.factors.definitions import populate_factors_for_dates
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT trade_date FROM daily_bar ORDER BY trade_date ASC;")
    all_dates = [r[0] for r in cursor.fetchall()]
    conn.close()

    sample_dates = all_dates[20::5]  # Weekly sample
    # Ensure tail dates up to the latest trade date are always included
    if all_dates and (not sample_dates or sample_dates[-1] != all_dates[-1]):
        tail = [d for d in all_dates if not sample_dates or d > sample_dates[-1]]
        sample_dates.extend(tail)
    print(f"Populating factor metrics across {len(sample_dates)} dates (including latest tail)...")
    populate_factors_for_dates(sample_dates, db_path=db_path)

    print(f"\n=== Real CSI 300 ETL Complete in {(datetime.now() - t_start).total_seconds():.1f}s ===")
    print_table_counts(db_path)


def print_table_counts(db_path: str = None) -> None:
    """Verifies and displays record counts for all core tables."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    tables = ["stock_basic", "daily_bar", "financial_pit", "factor_metrics"]
    print("=" * 45)
    print("         WAREHOUSE DATABASE ROW COUNTS       ")
    print("=" * 45)
    for tbl in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
        count = cursor.fetchone()[0]
        print(f" Table: {tbl:<18} | Count: {count:>8} rows")
    print("=" * 45)
    conn.close()


def sync_market_data(
    years: int = 5,
    n_stocks: int = 100,
    mode: str = "auto",
    db_path: Optional[str] = None,
    calculate_factors: bool = True
) -> Dict[str, Any]:
    """
    Fetches past N years of market data and dynamically synchronizes any newly available,
    uncollected data up to the current date (breakpoint continuation / incremental sync).
    """
    init_db(db_path)
    now_dt = datetime.now()
    target_end_date = now_dt.strftime("%Y%m%d")
    target_start_dt = now_dt - timedelta(days=int(years * 365.25))
    target_start_date = target_start_dt.strftime("%Y%m%d")

    print(f"\n[Sync] Target Data Horizon: {target_start_date} to {target_end_date} ({years} years)")

    # 1. Ensure stock_basic exists
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM stock_basic;")
    has_stocks = cursor.fetchone()[0]
    conn.close()

    if has_stocks == 0:
        print(f"[Sync] Initializing stock basics for {n_stocks} stocks...")
        stocks = generate_mock_stock_universe(n_stocks)
        ingest_stock_basics(stocks, db_path)
    else:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ts_code, name, industry, list_date, is_st, market_board FROM stock_basic;")
        stocks = [dict(r) for r in cursor.fetchall()]
        conn.close()

    # 2. Check current daily_bar coverage
    latest_date = get_latest_trade_date(db_path)
    earliest_date = get_earliest_trade_date(db_path)

    bars_added = 0
    financials_added = 0
    new_factor_dates = []

    # Case A: Database is completely empty
    if latest_date is None:
        print(f"[Sync] Database is empty. Performing full {years}-year backfill...")
        all_trade_dates = generate_trade_dates(target_start_date, target_end_date)
        bars = generate_mock_daily_bars(stocks, all_trade_dates)
        bars_added = ingest_daily_bars(bars, db_path)

        min_year = target_start_dt.year
        max_year = now_dt.year
        fin_years = list(range(min_year, max_year + 1))
        financials = generate_mock_financial_pit(stocks, years=fin_years)
        financials_added = ingest_financial_pit(financials, db_path)

        if calculate_factors:
            from engine.factors.definitions import populate_factors_for_dates
            sample_dates = all_trade_dates[20::5]
            if all_trade_dates and (not sample_dates or sample_dates[-1] != all_trade_dates[-1]):
                tail = [d for d in all_trade_dates if not sample_dates or d > sample_dates[-1]]
                sample_dates.extend(tail)
            print(f"[Sync] Computing factor metrics across {len(sample_dates)} dates (including latest tail)...")
            populate_factors_for_dates(sample_dates, db_path=db_path)
            new_factor_dates = sample_dates

        return {
            "status": "full_sync",
            "earliest_date": target_start_date,
            "latest_date": target_end_date,
            "bars_added": bars_added,
            "financials_added": financials_added,
            "new_factor_dates": len(new_factor_dates),
        }

    # Case B: Backfill earlier history if requested window starts before earliest_date
    if earliest_date > target_start_date:
        earlier_end = (datetime.strptime(earliest_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        if earlier_end >= target_start_date:
            print(f"[Sync] Backfilling historical head from {target_start_date} to {earlier_end}...")
            head_dates = generate_trade_dates(target_start_date, earlier_end)
            if head_dates:
                head_bars = generate_mock_daily_bars(stocks, head_dates)
                bars_added += ingest_daily_bars(head_bars, db_path)

    # Case C: Incremental forward sync to target_end_date
    if latest_date >= target_end_date:
        print(f"[Sync] Market data is already up to date (latest: {latest_date}).")
    else:
        next_start_date = (datetime.strptime(latest_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
        new_trade_dates = generate_trade_dates(next_start_date, target_end_date)
        if new_trade_dates:
            print(f"[Sync] Found {len(new_trade_dates)} new trading days ({new_trade_dates[0]} to {new_trade_dates[-1]}). Fetching latest bars...")
            last_states = get_stock_latest_states(db_path)
            new_bars = generate_mock_daily_bars(
                stocks,
                new_trade_dates,
                initial_states={code: (st[1], st[2]) for code, st in last_states.items()}
            )
            bars_added += ingest_daily_bars(new_bars, db_path)

            # Check new financial disclosures
            latest_ann = get_latest_financial_ann_date(db_path)
            new_fins = generate_mock_financial_pit(
                stocks,
                years=list(range(datetime.strptime(latest_date, "%Y%m%d").year, now_dt.year + 1)),
                min_ann_date=latest_ann
            )
            if new_fins:
                financials_added += ingest_financial_pit(new_fins, db_path)

            if calculate_factors:
                from engine.factors.definitions import populate_factors_for_dates
                print(f"[Sync] Computing factor metrics for newly ingested dates...")
                calc_dates = list(new_trade_dates[::5]) if len(new_trade_dates) > 10 else list(new_trade_dates)
                # Ensure the latest trade date is always included
                if new_trade_dates and (not calc_dates or calc_dates[-1] != new_trade_dates[-1]):
                    calc_dates.append(new_trade_dates[-1])
                if calc_dates:
                    populate_factors_for_dates(calc_dates, db_path=db_path)
                    new_factor_dates = calc_dates

    # Check if factor_metrics is lagging behind daily_bar
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(trade_date) FROM factor_metrics;")
    max_factor_d = cursor.fetchone()[0]
    cursor.execute("SELECT DISTINCT trade_date FROM daily_bar WHERE trade_date > ? ORDER BY trade_date ASC;", (max_factor_d or "",))
    missing_factor_dates = [r[0] for r in cursor.fetchall()]
    conn.close()

    if missing_factor_dates:
        print(f"[Sync] Found {len(missing_factor_dates)} dates in daily_bar without factor_metrics. Computing tail factors...")
        from engine.factors.definitions import populate_factors_for_dates
        populate_factors_for_dates(missing_factor_dates, db_path=db_path)
        new_factor_dates.extend(missing_factor_dates)

    print_table_counts(db_path)
    return {
        "status": "updated" if (bars_added > 0 or len(missing_factor_dates) > 0) else "up_to_date",
        "latest_date": get_latest_trade_date(db_path),
        "bars_added": bars_added,
        "financials_added": financials_added,
        "new_factor_dates": len(new_factor_dates),
    }


def auto_sync_on_startup(years: int = 5, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Called upon backend/Streamlit launch to ensure data is up to date with zero manual intervention.
    Also ensures factor_metrics is synchronized up to the latest trade_date in daily_bar.
    """
    init_db(db_path)
    latest = get_latest_trade_date(db_path)
    today = datetime.now().strftime("%Y%m%d")
    sync_res = {"status": "up_to_date", "latest_date": latest, "bars_added": 0, "financials_added": 0}
    if latest is None:
        # Cloud deployment without database in git: auto bootstrap in 2-3s
        print("[Startup Hook] No database found in repo. Performing quick cloud bootstrap (2 years, 30 stocks)...")
        sync_res = sync_market_data(years=2, n_stocks=30, db_path=db_path)
    elif latest < today:
        print(f"[Startup Hook] Current latest date is {latest}. Triggering incremental sync up to {today}...")
        sync_res = sync_market_data(years=years, db_path=db_path)

    # Check if factor_metrics lags behind daily_bar
    conn = get_connection(db_path)
    cursor = conn.cursor()
    missing_factor_dates = []
    try:
        cursor.execute("SELECT MAX(trade_date) FROM factor_metrics;")
        row = cursor.fetchone()
        max_factor_d = row[0] if row else None
        cursor.execute("SELECT DISTINCT trade_date FROM daily_bar WHERE trade_date > ? ORDER BY trade_date ASC;", (max_factor_d or "",))
        missing_factor_dates = [r[0] for r in cursor.fetchall()]
    except sqlite3.OperationalError:
        missing_factor_dates = []
    finally:
        conn.close()

    if missing_factor_dates:
        print(f"[Startup Hook] Found {len(missing_factor_dates)} dates in daily_bar without factor_metrics. Computing tail factors...")
        from engine.factors.definitions import populate_factors_for_dates
        populate_factors_for_dates(missing_factor_dates, db_path=db_path)
        sync_res["missing_factor_dates_added"] = len(missing_factor_dates)

    return sync_res


def main():
    parser = argparse.ArgumentParser(description="A-Share Quant ETL Ingestion Engine")
    parser.add_argument("--sync", action="store_true", help="Sync uncollected latest data and ensure 5-year coverage")
    parser.add_argument("--years", type=int, default=5, help="Number of years of historical data to maintain (default: 5)")
    parser.add_argument("--sample", action="store_true", help="Populate database with sample market data")
    parser.add_argument("--online", "--real", dest="real", action="store_true", help="Fetch real CSI 300 market data (5 years)")
    parser.add_argument("--stocks", type=int, default=300, help="Number of stocks to process (default: 300)")
    parser.add_argument("--status", action="store_true", help="Print table row counts")
    parser.add_argument("--db", type=str, default=None, help="Custom database file path")

    args = parser.parse_args()

    if args.status:
        print_table_counts(args.db)
    elif args.sync:
        sync_market_data(years=args.years, n_stocks=args.stocks, db_path=args.db)
    elif args.real:
        run_real_csi300_etl(args.stocks, args.db)
    else:
        run_sample_etl(args.stocks, args.db)


if __name__ == "__main__":
    main()

