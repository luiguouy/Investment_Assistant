# -*- coding: utf-8 -*-
"""
智能股票技术分析平台 - 本地 K线数据与图表标注缓存层 (SQLite WAL)

目标:
  把每个 (symbol, period) 拉取过的全量历史以及图表标注持久化到本地 SQLite 数据库，
  开启 WAL 模式保证高并发低 I/O，存储路径默认采用 %APPDATA%/StockVision/stock_data.db。
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any

import pandas as pd

# 优先存放在 %APPDATA%/StockVision/
app_data = os.environ.get("APPDATA")
if app_data:
    SV_DIR = os.path.join(app_data, "StockVision")
    os.makedirs(SV_DIR, exist_ok=True)
    DB_PATH = os.path.join(SV_DIR, "stock_data.db")
else:
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data.db")

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS klines (
        symbol TEXT NOT NULL,
        period TEXT NOT NULL,
        date   TEXT NOT NULL,
        open   REAL,
        high   REAL,
        low    REAL,
        close  REAL,
        volume REAL,
        PRIMARY KEY (symbol, period, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        symbol       TEXT NOT NULL,
        period       TEXT NOT NULL,
        last_updated TEXT,
        source       TEXT,
        PRIMARY KEY (symbol, period)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_klines_sym_period ON klines(symbol, period)",
    """
    CREATE TABLE IF NOT EXISTS annotations (
        id         TEXT PRIMARY KEY,
        symbol     TEXT NOT NULL,
        kind       TEXT NOT NULL,
        data       TEXT NOT NULL,
        source     TEXT,
        created_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_anno_sym ON annotations(symbol, kind)",
]


_tables_initialized = False


def _init_journal_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            period TEXT NOT NULL DEFAULT '1d',
            side TEXT NOT NULL,                      -- 'LONG' | 'SHORT'
            status TEXT NOT NULL DEFAULT 'OPEN',     -- 'OPEN' | 'CLOSED'
            entry_time TEXT NOT NULL,                -- YYYY-MM-DD 或 YYYY-MM-DD HH:MM
            entry_price REAL NOT NULL,
            shares REAL NOT NULL DEFAULT 0,
            planned_stop REAL,                       -- 系统建议止损价
            planned_tp1 REAL,                        -- 系统建议 TP1
            planned_rr REAL,                         -- 系统规划盈亏比
            exit_time TEXT,
            exit_price REAL,
            pnl_pct REAL,                            -- 实际收益率 (%)
            realized_r REAL,                         -- 实际实现 R 倍数
            -- 结构化归因标签（从 summary 提取，便于 SQL 直接分组统计）
            wave_pattern TEXT DEFAULT '',
            dow_phase TEXT DEFAULT '',
            dow_market_status TEXT DEFAULT '',       -- PASS / DIVERGENCE / BEAR / BREADTH_DIVERGENCE
            passed_gates INTEGER DEFAULT 0,          -- 下单时是否通过系统风控硬闸门 (0/1)
            discipline_violation TEXT DEFAULT '',    -- 纪律偏离标记（如 '扛单超损' / '逆大盘强开'）
            notes TEXT DEFAULT '',
            -- 完整现场还原快照（JSON）
            summary_json TEXT,
            annotations_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_symbol ON trade_journal(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_status ON trade_journal(status)")


def _get_connection() -> sqlite3.Connection:
    """获取启用了 WAL 模式的高性能 SQLite 连接，并确保表结构已初始化。"""
    global _tables_initialized
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    if not _tables_initialized:
        cur = conn.cursor()
        for stmt in _SCHEMA:
            cur.execute(stmt)
        _init_journal_table(conn)
        conn.commit()
        _tables_initialized = True
    return conn


def init_db() -> None:
    """初始化数据库表结构 (幂等, 可重复调用)。"""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        for stmt in _SCHEMA:
            cur.execute(stmt)
        _init_journal_table(conn)
        conn.commit()
    finally:
        conn.close()


def get_cached_klines(
    symbol: str, period: str, max_age_hours: Union[int, float] = 24
) -> Optional[List[Dict]]:
    """
    读取缓存的 K线数据。
    """
    from data_provider import validate_klines

    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT last_updated FROM meta WHERE symbol=? AND period=?",
            (symbol, period),
        )
        row = cur.fetchone()
        if not row:
            return None

        try:
            last_updated = datetime.fromisoformat(row[0])
            if datetime.now() - last_updated > timedelta(hours=max_age_hours):
                return None  # 过期
        except Exception:
            return None

        cur.execute(
            "SELECT date, open, high, low, close, volume "
            "FROM klines WHERE symbol=? AND period=? "
            "ORDER BY date ASC",
            (symbol, period),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        data = [
            {
                "date": r[0],
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
            }
            for r in rows
        ]

        df_check = pd.DataFrame(data)
        if not validate_klines(df_check, period):
            return None

        return data
    finally:
        conn.close()


def save_klines(
    symbol: str, period: str, klines: List[Dict], source: str = ""
) -> None:
    """将 K线数据批量写入缓存 (先清空旧数据再插入, 保证一致性)。"""
    if not klines:
        return
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM klines WHERE symbol=? AND period=?", (symbol, period)
        )
        cur.executemany(
            "INSERT OR REPLACE INTO klines "
            "(symbol, period, date, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    symbol,
                    period,
                    k["date"],
                    k["open"],
                    k["high"],
                    k["low"],
                    k["close"],
                    k["volume"],
                )
                for k in klines
            ],
        )
        cur.execute(
            "INSERT OR REPLACE INTO meta (symbol, period, last_updated, source) "
            "VALUES (?,?,?,?)",
            (symbol, period, datetime.now().isoformat(), source),
        )
        conn.commit()
    finally:
        conn.close()


def cache_status(symbol: str, period: str) -> Optional[Dict]:
    """查询某 (symbol, period) 的缓存元信息 (调试用)。"""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT last_updated, source FROM meta WHERE symbol=? AND period=?",
            (symbol, period),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"last_updated": row[0], "source": row[1]}
    finally:
        conn.close()


# ============================================================
# 图表标注持久化接口 (Markers / PriceLines / Boxes)
# ============================================================

def save_annotation(anno_id: str, symbol: str, kind: str, data: dict, source: str = "") -> None:
    """保存或更新单个图表标注。"""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        now_str = datetime.now().isoformat()
        cur.execute(
            "INSERT OR REPLACE INTO annotations (id, symbol, kind, data, source, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (anno_id, symbol.upper(), kind, json.dumps(data, ensure_ascii=False), source, now_str),
        )
        conn.commit()
    finally:
        conn.close()


def save_annotations_batch(symbol: str, kind: str, items: List[dict], source: str = "") -> None:
    """批量保存图表标注。"""
    if not items:
        return
    conn = _get_connection()
    try:
        cur = conn.cursor()
        now_str = datetime.now().isoformat()
        records = []
        for it in items:
            aid = str(it.get("id") or f"{kind}_{symbol}_{datetime.now().timestamp()}_{len(records)}")
            records.append((aid, symbol.upper(), kind, json.dumps(it, ensure_ascii=False), source, now_str))
        cur.executemany(
            "INSERT OR REPLACE INTO annotations (id, symbol, kind, data, source, created_at) "
            "VALUES (?,?,?,?,?,?)",
            records,
        )
        conn.commit()
    finally:
        conn.close()


def get_annotations(symbol: str, kind: Optional[str] = None) -> List[dict]:
    """获取指定股票的所有标注。"""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        if kind:
            cur.execute(
                "SELECT id, symbol, kind, data, source, created_at FROM annotations "
                "WHERE symbol=? AND kind=? ORDER BY created_at ASC",
                (symbol.upper(), kind),
            )
        else:
            cur.execute(
                "SELECT id, symbol, kind, data, source, created_at FROM annotations "
                "WHERE symbol=? ORDER BY created_at ASC",
                (symbol.upper(),),
            )
        rows = cur.fetchall()
        result = []
        for r in rows:
            try:
                payload = json.loads(r[3])
                payload["id"] = r[0]
                payload["symbol"] = r[1]
                payload["kind"] = r[2]
                payload["source"] = r[4]
                payload["created_at"] = r[5]
                result.append(payload)
            except Exception:
                continue
        return result
    finally:
        conn.close()


def clear_annotations(symbol: str, kind: Optional[str] = None, source: Optional[str] = None) -> int:
    """按条件清理标注。返回被删除的条数。"""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        sql = "DELETE FROM annotations WHERE symbol=?"
        params = [symbol.upper()]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        if source:
            sql += " AND source=?"
            params.append(source)
        cur.execute(sql, params)
        deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def create_journal_entry(data: Dict[str, Any]) -> int:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO trade_journal (
                symbol, period, side, status, entry_time, entry_price, shares,
                planned_stop, planned_tp1, planned_rr,
                wave_pattern, dow_phase, dow_market_status, passed_gates,
                discipline_violation, notes, summary_json, annotations_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["symbol"],
                data.get("period", "1d"),
                data["side"],
                data["entry_time"],
                float(data["entry_price"]),
                float(data.get("shares") or 0),
                data.get("planned_stop"),
                data.get("planned_tp1"),
                data.get("planned_rr"),
                data.get("wave_pattern", ""),
                data.get("dow_phase", ""),
                data.get("dow_market_status", ""),
                1 if data.get("passed_gates") else 0,
                data.get("discipline_violation", ""),
                data.get("notes", ""),
                json.dumps(data.get("summary") or {}, ensure_ascii=False),
                json.dumps(data.get("annotations") or {}, ensure_ascii=False),
                now_str,
                now_str,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def close_journal_entry(trade_id: int, exit_price: float, exit_time: str, notes: Optional[str] = None) -> Optional[Dict[str, Any]]:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM trade_journal WHERE id = ?", (trade_id,)).fetchone()
        if not row:
            return None

        entry_p = float(row["entry_price"])
        side = str(row["side"]).upper()
        sign = 1.0 if side == "LONG" else -1.0
        pnl_pct = round(((exit_price - entry_p) / max(entry_p, 1e-6)) * sign * 100.0, 2)

        planned_stop = row["planned_stop"]
        realized_r = None
        violations = [v for v in (row["discipline_violation"] or "").split(",") if v]

        if planned_stop is not None and abs(entry_p - float(planned_stop)) > 1e-6:
            risk = abs(entry_p - float(planned_stop))
            realized_r = round(((exit_price - entry_p) * sign) / risk, 2)
            # 纪律审计：实际亏损超过规划止损 1.25R 以上，自动标记「扛单超损」
            if realized_r < -1.25 and "扛单超损" not in violations:
                violations.append("扛单超损")

        merged_notes = row["notes"] or ""
        if notes:
            merged_notes = f"{merged_notes}\n[平仓] {notes}".strip()

        conn.execute(
            """
            UPDATE trade_journal
            SET status = 'CLOSED', exit_time = ?, exit_price = ?,
                pnl_pct = ?, realized_r = ?, discipline_violation = ?,
                notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (exit_time, float(exit_price), pnl_pct, realized_r, ",".join(violations), merged_notes, now_str, trade_id),
        )
        conn.commit()
        return get_journal_detail(trade_id)
    finally:
        conn.close()


def list_journal_entries(symbol: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    try:
        sql = """
            SELECT id, symbol, period, side, status, entry_time, entry_price, shares,
                   planned_stop, planned_tp1, planned_rr, exit_time, exit_price,
                   pnl_pct, realized_r, wave_pattern, dow_phase, dow_market_status,
                   passed_gates, discipline_violation, notes, created_at
            FROM trade_journal WHERE 1=1
        """
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if status:
            sql += " AND status = ?"
            params.append(status.upper())
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_journal_detail(trade_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM trade_journal WHERE id = ?", (trade_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        d["annotations"] = json.loads(d.pop("annotations_json") or "{}")
        return d
    finally:
        conn.close()


def delete_journal_entry(trade_id: int) -> bool:
    conn = _get_connection()
    try:
        cur = conn.execute("DELETE FROM trade_journal WHERE id = ?", (trade_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_journal_attribution() -> Dict[str, Any]:
    """量化归因统计：按浪型、大盘印证状态、是否遵守风控门禁聚合真实胜率与平均 R。"""
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM trade_journal WHERE status = 'CLOSED'").fetchall()]
        total = len(rows)
        if total == 0:
            return {
                "total_closed": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "avg_r": 0.0,
                "by_wave": [], "by_market_status": [], "by_gate_compliance": [], "violation_stats": {},
            }

        def _agg(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
            n = len(subset)
            wins = sum(1 for x in subset if (x.get("pnl_pct") or 0) > 0)
            avg_pnl = round(sum(float(x.get("pnl_pct") or 0) for x in subset) / n, 2)
            r_vals = [float(x["realized_r"]) for x in subset if x.get("realized_r") is not None]
            avg_r = round(sum(r_vals) / len(r_vals), 2) if r_vals else 0.0
            return {"count": n, "win_rate": round(wins / n * 100.0, 1), "avg_pnl_pct": avg_pnl, "avg_r": avg_r}

        def _group_by(field: str, fallback: str = "未分类") -> List[Dict[str, Any]]:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                k = str(r.get(field) or fallback).strip() or fallback
                buckets.setdefault(k, []).append(r)
            return sorted(
                [{"label": k, **_agg(v)} for k, v in buckets.items()],
                key=lambda x: x["count"],
                reverse=True,
            )

        gate_yes = [r for r in rows if r.get("passed_gates")]
        gate_no = [r for r in rows if not r.get("passed_gates")]
        by_gate = []
        if gate_yes:
            by_gate.append({"label": "符合风控硬闸门", **_agg(gate_yes)})
        if gate_no:
            by_gate.append({"label": "违背闸门强行开仓", **_agg(gate_no)})

        v_counts: Dict[str, int] = {}
        for r in rows:
            for v in (r.get("discipline_violation") or "").split(","):
                v = v.strip()
                if v:
                    v_counts[v] = v_counts.get(v, 0) + 1

        overall = _agg(rows)
        return {
            "total_closed": total,
            **overall,
            "by_wave": _group_by("wave_pattern"),
            "by_market_status": _group_by("dow_market_status", "未记录"),
            "by_gate_compliance": by_gate,
            "violation_stats": v_counts,
        }
    finally:
        conn.close()
