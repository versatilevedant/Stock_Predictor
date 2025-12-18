# main_db_integration_example.py
import tkinter as tk
from tkinter import ttk, messagebox
from database import init_db, df_to_db, list_symbols, load_symbol, delete_symbol
from file_manager import import_csv_via_dialog, save_symbol_csv_via_dialog, save_full_db_csv_via_dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize DB
init_db()

# Simple GUI
root = tk.Tk()
root.title("DB & File Handling - Example")
root.geometry("900x600")

top_frame = ttk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

symbol_label = ttk.Label(top_frame, text="Symbol:")
symbol_label.pack(side=tk.LEFT)
symbol_var = tk.StringVar()
symbol_combo = ttk.Combobox(top_frame, textvariable=symbol_var, values=list_symbols(), width=12)
symbol_combo.pack(side=tk.LEFT, padx=6)

def refresh_symbol_list():
    symbol_combo['values'] = list_symbols()

def on_import_csv():
    df, symbol = import_csv_via_dialog()
    if df is None:
        return
    # store into DB
    count = df_to_db(df, symbol)
    messagebox.showinfo("Imported", f"Imported records (attempted): {count}")
    refresh_symbol_list()

def on_export_symbol():
    sym = symbol_var.get().strip().upper()
    if not sym:
        messagebox.showwarning("Select symbol", "Choose a symbol first.")
        return
    path = save_symbol_csv_via_dialog(sym)
    if path:
        messagebox.showinfo("Exported", f"Exported {sym} to {path}")

def on_export_all():
    path = save_full_db_csv_via_dialog()
    if path:
        messagebox.showinfo("Exported", f"Exported DB to {path}")

def on_delete_symbol():
    sym = symbol_var.get().strip().upper()
    if not sym:
        messagebox.showwarning("Select symbol", "Choose a symbol first.")
        return
    confirmed = messagebox.askyesno("Confirm", f"Delete all records for {sym}?")
    if not confirmed:
        return
    deleted = delete_symbol(sym)
    messagebox.showinfo("Deleted", f"Deleted {deleted} rows for {sym}.")
    refresh_symbol_list()

def on_plot():
    for w in plot_frame.winfo_children():
        w.destroy()
    sym = symbol_var.get().strip().upper()
    if not sym:
        messagebox.showwarning("Select symbol", "Choose a symbol first.")
        return
    df = load_symbol(sym)
    if df.empty:
        messagebox.showinfo("No data", f"No rows for {sym} in DB.")
        return

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df['date'], df['close'], linewidth=2)
    ax.set_title(f"{sym} Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    fig.autofmt_xdate()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

btn_import = ttk.Button(top_frame, text="Import CSV → DB", command=on_import_csv)
btn_import.pack(side=tk.LEFT, padx=6)
btn_export_sym = ttk.Button(top_frame, text="Export Symbol → CSV", command=on_export_symbol)
btn_export_sym.pack(side=tk.LEFT, padx=6)
btn_export_all = ttk.Button(top_frame, text="Export DB → CSV", command=on_export_all)
btn_export_all.pack(side=tk.LEFT, padx=6)
btn_delete = ttk.Button(top_frame, text="Delete Symbol", command=on_delete_symbol)
btn_delete.pack(side=tk.LEFT, padx=6)
btn_plot = ttk.Button(top_frame, text="Plot Symbol", command=on_plot)
btn_plot.pack(side=tk.LEFT, padx=6)

plot_frame = ttk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
