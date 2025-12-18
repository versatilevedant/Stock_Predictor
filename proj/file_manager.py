# file_manager.py
import pandas as pd
from tkinter import filedialog, messagebox
import os
from database import df_to_db, export_symbol_to_csv, export_db_to_csv

def import_csv_via_dialog(symbol_hint: str = None):
    """
    Open a file dialog, let user pick a CSV, read it with pandas, return (df, path).
    Caller should pass df to df_to_db() to persist.
    """
    filepath = filedialog.askopenfilename(
        title="Select stock CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not filepath:
        return None, None

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV: {e}")
        return None, None

    # Try to guess symbol from filename if not provided
    symbol = symbol_hint
    if not symbol:
        base = os.path.basename(filepath)
        symbol = os.path.splitext(base)[0].upper()
    return df, symbol


def save_symbol_csv_via_dialog(symbol: str):
    path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile=f"{symbol}.csv",
        title="Export symbol to CSV"
    )
    if not path:
        return None
    export_symbol_to_csv(symbol, path)
    return path


def save_full_db_csv_via_dialog():
    path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="all_stock_data.csv",
        title="Export full DB to CSV"
    )
    if not path:
        return None
    export_db_to_csv(path)
    return path
