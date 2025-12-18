# database.py
import sqlite3
from typing import Optional
import pandas as pd
import os
from config import DB_PATH


def init_db(db_path: str = DB_PATH):
    """Create the database and the stock_history table with a UNIQUE constraint on (symbol, date)."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS stock_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        adj_close REAL,
        volume INTEGER,
        UNIQUE(symbol, date) ON CONFLICT REPLACE
    )
    """)
    conn.commit()
    conn.close()


def df_to_db(df: pd.DataFrame, symbol: str, db_path: str = DB_PATH):
    """
    Insert a DataFrame into stock_history.
    The df is expected to have a 'Date' or 'date' column and numeric OHLCV columns.
    Uses pandas.to_sql for speed, but ensures the final table conforms to schema and dedup via UNIQUE constraint.
    """
    if df.empty:
        return 0

    # Normalize column names
    df = df.copy()
    def _to_str_lower(c):
        if isinstance(c, str):
            return c.lower()
        if isinstance(c, tuple) and len(c) > 0:
            # Prefer first stringy component
            for part in c:
                if isinstance(part, str) and part:
                    return part.lower()
            return str(c).lower()
        return str(c).lower()
    df.columns = [_to_str_lower(c) for c in df.columns]

    # Ensure 'date' column exists
    if 'date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'date'}, inplace=True)
    if 'date' not in df.columns:
        # Try to detect datetime index
        if getattr(df, 'index', None) is not None:
            df = df.reset_index()
            if 'index' in df.columns:
                df.rename(columns={'index': 'date'}, inplace=True)
    # Coerce date col to ISO string (YYYY-MM-DD)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Standardize column names to match DB
    col_map = {}
    for c in ['open', 'high', 'low', 'close', 'adj close', 'adj_close', 'volume']:
        if c in df.columns:
            # map 'adj close' -> 'adj_close'
            col_map[c] = c.replace(' ', '_')
    df.rename(columns=col_map, inplace=True)

    # Keep only columns we want and ensure all exist (fill missing with NULLs)
    allowed = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in allowed:
        if col not in df.columns:
            df[col] = pd.NA
    insert_df = df[allowed].copy()
    insert_df['symbol'] = symbol

    # Write to a temporary table then insert with INSERT OR REPLACE to keep schema control.
    conn = sqlite3.connect(db_path)
    try:
        insert_df.to_sql("tmp_import", conn, if_exists="replace", index=False)
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO stock_history (symbol, date, open, high, low, close, adj_close, volume)
            SELECT symbol, date,
                   COALESCE(open, NULL),
                   COALESCE(high, NULL),
                   COALESCE(low, NULL),
                   COALESCE(close, NULL),
                   COALESCE(adj_close, NULL),
                   COALESCE(volume, NULL)
            FROM tmp_import
        """)
        conn.commit()
        # clean up temp table
        cur.execute("DROP TABLE IF EXISTS tmp_import")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def load_symbol(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, db_path: str = DB_PATH) -> pd.DataFrame:
    """Load rows for a symbol (optionally between start_date and end_date). Dates should be 'YYYY-MM-DD' strings."""
    conn = sqlite3.connect(db_path)
    try:
        q = "SELECT symbol, date, open, high, low, close, adj_close, volume FROM stock_history WHERE symbol = ?"
        params = [symbol]
        if start_date:
            q += " AND date >= ?"
            params.append(start_date)
        if end_date:
            q += " AND date <= ?"
            params.append(end_date)
        q += " ORDER BY date ASC"
        df = pd.read_sql(q, conn, params=params, parse_dates=['date'])
        return df
    finally:
        conn.close()


def list_symbols(db_path: str = DB_PATH):
    """Return a list of symbols stored in DB."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM stock_history ORDER BY symbol")
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def delete_symbol(symbol: str, db_path: str = DB_PATH):
    """Delete all records for a symbol."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM stock_history WHERE symbol = ?", (symbol,))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def export_symbol_to_csv(symbol: str, csv_path: str, db_path: str = DB_PATH):
    df = load_symbol(symbol, db_path=db_path)
    df.to_csv(csv_path, index=False)
    return csv_path


def export_db_to_csv(csv_path: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM stock_history ORDER BY symbol, date", conn, parse_dates=['date'])
        df.to_csv(csv_path, index=False)
        return csv_path
    finally:
        conn.close()


def export_symbol_to_json(symbol: str, json_path: str, db_path: str = DB_PATH):
    df = load_symbol(symbol, db_path=db_path)
    df.to_json(json_path, orient="records", date_format='iso')
    return json_path


def backup_db(backup_path: str, db_path: str = DB_PATH):
    """Make a file copy of the sqlite DB file."""
    import shutil
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(db_path, backup_path)
    return backup_path


def restore_db(backup_path: str, db_path: str = DB_PATH):
    """Restore DB from backup (overwrites current DB)."""
    import shutil
    shutil.copy2(backup_path, db_path)
    return db_path
