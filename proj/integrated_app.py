"""
integrated_app.py
-----------------
Complete Stock Market Predictor with ML, Database, and Comprehensive GUI
Integrates all components: frontend, ML predictor, database, and visualizations
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import local modules
from database import init_db, df_to_db, list_symbols, load_symbol, delete_symbol, export_symbol_to_csv
from config import *
import joblib
import os

# Import ML components
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
# XGBoost is optional; if it fails to load (e.g., libomp), we continue without it
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False
from statsmodels.tsa.arima.model import ARIMA

# Initialize database
init_db()


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure yfinance DataFrame has single-level string columns (e.g., 'Close')."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    else:
        # Also handle tuple column names defensively
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple) and len(c) > 0:
                new_cols.append(c[0] if isinstance(c[0], str) and c[0] else str(c))
            else:
                new_cols.append(c)
        df = df.copy()
        df.columns = new_cols
    return df


class StockPredictorApp:

    """Comprehensive Stock Market Predictor Application"""

    def refresh_db_view(self):
        """Refresh database treeview with records for current symbol"""
        try:
            # Clear existing items
            for item in self.db_tree.get_children():
                self.db_tree.delete(item)
            
            # Get current symbol
            symbol = self.current_symbol.get().strip().upper()
            if not symbol:
                self.update_results("⚠️ Enter a stock symbol to view its database records")
                return
            
            # Load data for current symbol
            df = load_symbol(symbol)
            if df.empty:
                self.update_results(f"⚠️ No records found for {symbol} in database")
                return
            
            # Show all records for the symbol
            for _, row in df.iterrows():
                self.db_tree.insert("", "end", values=(
                    symbol,
                    row.get('date', ''),
                    f"{row.get('close', 0):.2f}",
                    row.get('volume', 0)
                ))
            
            self.update_results(f"✅ Showing {len(df)} records for {symbol}")
        except Exception as e:
            messagebox.showerror("Refresh Error", str(e))

    def load_from_db(self):
        """Load stock data from database for current symbol"""
        symbol = self.current_symbol.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol")
            return
        
        try:
            df = load_symbol(symbol)
            if df.empty:
                messagebox.showinfo("No Data", f"No data found for {symbol} in database.")
                return
            
            self.update_results(f"📥 Loaded {len(df)} records for {symbol} from database")
            messagebox.showinfo("Success", f"Loaded {len(df)} records for {symbol}")
            
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
    
    def __init__(self, root):
        self.root = root
        self.root.title("💹 Stock Market Predictor - Integrated Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLOR_BG_DARK)
        
        # State variables
        self.current_symbol = tk.StringVar(value=DEFAULT_SYMBOL)
        self.is_training = False
        self.models_loaded = False
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()

        
    def _resolve_symbol(self, sym: str) -> str:
        """Resolve user input to a valid Yahoo Finance symbol.
        Tries as-typed, then '.NS', then '.BO'. Returns the first that yields data.
        """
        s = sym.strip().upper()
        if not s:
            raise ValueError("Please enter a stock symbol")
        candidates = [s]
        if "." not in s:  # no exchange suffix provided
            candidates += [f"{s}.NS", f"{s}.BO"]
        for cand in candidates:
            try:
                df_test = yf.download(cand, period="5d", interval="1d", progress=False)
                if not df_test.empty:
                    return cand
            except Exception:
                continue
        # Fallback to original; let downstream show data error
        return s
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("TLabel", background=COLOR_BG_DARK, foreground=COLOR_TEXT_LIGHT, 
                       font=("Inter", 10))
        style.configure("Header.TLabel", font=("Inter", 20, "bold"), 
                       foreground=COLOR_ACCENT_TEAL)
        style.configure("Section.TLabel", font=("Inter", 12, "bold"), 
                       foreground=COLOR_ACCENT_TEAL)
        style.configure("Result.TLabel", font=("Inter", 14, "bold"), 
                       foreground=COLOR_PREDICTION_GOLD)
        
        style.configure("TButton", font=("Inter", 10, "bold"), padding=8,
                       background=COLOR_ACCENT_TEAL, foreground=COLOR_TEXT_DARK)
        style.map("TButton", background=[("active", "#4DB6AC")])
        
        style.configure("TEntry", fieldbackground=COLOR_BG_MEDIUM, 
                       foreground=COLOR_TEXT_LIGHT, insertcolor=COLOR_ACCENT_TEAL)
        style.configure("TFrame", background=COLOR_BG_MEDIUM)
        style.configure("Card.TFrame", background=COLOR_BG_MEDIUM, relief="groove", 
                       borderwidth=2)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=(15, 10), fill="x", padx=20)
        
        header = ttk.Label(header_frame, 
                          text="📈 STOCK MARKET PREDICTOR - INTEGRATED DASHBOARD",
                          style="Header.TLabel", anchor="center")
        header.pack(expand=True, fill="x")
        
        # Create footer first to ensure proper packing order
        self._create_footer()
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(padx=20, pady=10, fill="both", expand=True)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=2)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Left panel
        left_panel = self.create_left_panel(main_container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Right panel (tabbed)
        right_panel = self.create_right_panel(main_container)
        right_panel.grid(row=0, column=1, sticky="nsew")

    def _create_footer(self):
        # Remove any existing footer frame (e.g., after theme refresh)
        if hasattr(self, "footer_frame") and self.footer_frame.winfo_exists():
            self.footer_frame.destroy()

        self.footer_frame = tk.Frame(self.root, bg=COLOR_BG_DARK, height=35)
        self.footer_frame.pack(side="bottom", fill="x")
        self.footer_frame.pack_propagate(False)

        watermark = tk.Label(
            self.footer_frame,
            text="© Made By Team OptiML",
            bg=COLOR_BG_DARK,
            fg=COLOR_ACCENT_TEAL,
            font=("Inter", 11, "bold"),
            anchor="center",
        )
        watermark.pack(expand=True, fill="both")
        
    def create_left_panel(self, parent):
        """Create left control panel"""
        panel = ttk.Frame(parent, style="TFrame")
        
        # Input card
        input_card = ttk.Frame(panel, style="Card.TFrame")
        input_card.pack(pady=(0, 15), fill="x")
        
        ttk.Label(input_card, text="Stock Symbol:", style="Section.TLabel",
                 background=COLOR_BG_MEDIUM).pack(pady=(12, 5), padx=12, anchor="w")
        
        symbol_frame = ttk.Frame(input_card, style="Card.TFrame")
        symbol_frame.pack(pady=5, padx=12, fill="x")
        
        self.symbol_entry = ttk.Entry(symbol_frame, textvariable=self.current_symbol,
                                     font=("Inter", 11), width=20)
        self.symbol_entry.pack(side="left", fill="x", expand=True, padx=(4, 5))
        if not self.current_symbol.get():
            self.symbol_entry.insert(0, DEFAULT_SYMBOL)
        
        ttk.Button(symbol_frame, text="📥", command=self.load_from_db,
                  width=3).pack(side="left")
        
        # ML Actions
        ttk.Label(input_card, text="ML Actions:", style="Section.TLabel",
                 background=COLOR_BG_MEDIUM).pack(pady=(10, 5), padx=12, anchor="w")
        
        btn_train = ttk.Button(input_card, text="🎓 Train Models", 
                              command=self.train_models)
        btn_train.pack(pady=5, padx=12, fill="x")
        
        btn_predict = ttk.Button(input_card, text="🔮 Predict Next Day",
                                command=self.predict_next_day)
        btn_predict.pack(pady=5, padx=12, fill="x")
        
        btn_forecast = ttk.Button(input_card, text="📊 30-Day Forecast",
                                 command=self.hybrid_forecast)
        btn_forecast.pack(pady=5, padx=12, fill="x")
        
        # Database Actions
        ttk.Label(input_card, text="Database Actions:", style="Section.TLabel",
                 background=COLOR_BG_MEDIUM).pack(pady=(10, 5), padx=12, anchor="w")
        
        btn_import = ttk.Button(input_card, text="📂 Import CSV to DB",
                               command=self.import_csv_to_db)
        btn_import.pack(pady=5, padx=12, fill="x")
        
        btn_export = ttk.Button(input_card, text="💾 Export Symbol to CSV",
                               command=self.export_symbol_csv)
        btn_export.pack(pady=5, padx=12, fill="x")
        
        btn_delete = ttk.Button(input_card, text="🗑️ Delete Symbol from DB",
                               command=self.delete_symbol_from_db)
        btn_delete.pack(pady=5, padx=12, fill="x")
        
        btn_list = ttk.Button(input_card, text="📋 List All Symbols",
                             command=self.list_all_symbols)
        btn_list.pack(pady=(5, 12), padx=12, fill="x")
        
        # Results card
        results_card = ttk.Frame(panel, style="Card.TFrame")
        results_card.pack(pady=0, fill="both", expand=True)
        
        ttk.Label(results_card, text="Results:", style="Section.TLabel",
                 background=COLOR_BG_MEDIUM).pack(pady=(12, 8), padx=12, anchor="w")
        
        # Results text widget with scrollbar
        results_frame = ttk.Frame(results_card, style="Card.TFrame")
        results_frame.pack(pady=5, padx=12, fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.results_text = tk.Text(results_frame, height=20, width=35,
                                   bg=COLOR_BG_DARK, fg=COLOR_TEXT_LIGHT,
                                   font=("Courier", 10), wrap="word",
                                   yscrollcommand=scrollbar.set)
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        return panel
        
    def create_right_panel(self, parent):
        """Create right panel with tabs"""
        panel = ttk.Frame(parent, style="Card.TFrame")
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(panel)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Price Chart
        self.chart_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.chart_tab, text="📈 Price Chart")
        self.create_chart_tab()
        
        # Tab 2: Analytics
        self.analytics_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.analytics_tab, text="📊 Analytics")
        self.create_analytics_tab()
        
        # Tab 3: Database View
        self.db_tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(self.db_tab, text="🗄️ Database")
        self.create_db_tab()
        
        return panel
        
    def create_chart_tab(self):
        """Create price chart visualization"""
        ttk.Label(self.chart_tab, text="Stock Price Trend",
                 style="Section.TLabel", background=COLOR_BG_MEDIUM).pack(
                     pady=(10, 5), padx=10, anchor="w")
        
        # Matplotlib figure
        self.fig_price = Figure(figsize=(10, 6), dpi=100)
        self.ax_price = self.fig_price.add_subplot(111)
        self.fig_price.patch.set_facecolor(COLOR_BG_MEDIUM)
        self.ax_price.set_facecolor(COLOR_BG_MEDIUM)
        
        self.canvas_price = FigureCanvasTkAgg(self.fig_price, master=self.chart_tab)
        self.canvas_price.get_tk_widget().pack(pady=5, padx=10, fill="both", expand=True)
        
        self.plot_initial_chart()
        
    def create_analytics_tab(self):
        """Create analytics visualization with multiple charts"""
        # Header
        header_frame = ttk.Frame(self.analytics_tab, style="TFrame")
        header_frame.pack(fill="x", pady=(10, 5), padx=10)
        
        ttk.Label(header_frame, text="Comprehensive Analytics",
                 style="Section.TLabel", background=COLOR_BG_MEDIUM).pack(side="left")
        
        ttk.Button(header_frame, text="🔄 Generate Analytics",
                  command=self.generate_analytics).pack(side="right", padx=5)
        
        # Create canvas with scrollbar for multiple charts
        canvas_container = tk.Canvas(self.analytics_tab, bg=COLOR_BG_MEDIUM, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.analytics_tab, orient="vertical", command=canvas_container.yview)
        self.analytics_scroll_frame = ttk.Frame(canvas_container, style="TFrame")
        
        self.analytics_scroll_frame.bind(
            "<Configure>",
            lambda e: canvas_container.configure(scrollregion=canvas_container.bbox("all"))
        )
        
        canvas_container.create_window((0, 0), window=self.analytics_scroll_frame, anchor="nw")
        canvas_container.configure(yscrollcommand=scrollbar.set)
        
        canvas_container.pack(side="left", fill="both", expand=True, padx=10, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Initial message
        ttk.Label(self.analytics_scroll_frame, 
                 text="Click 'Generate Analytics' to view comprehensive charts",
                 style="Section.TLabel", background=COLOR_BG_MEDIUM).pack(pady=50)
        
    def create_db_tab(self):
        """Create database view"""
        ttk.Label(self.db_tab, text="Database Content",
                 style="Section.TLabel", background=COLOR_BG_MEDIUM).pack(
                     pady=(10, 5), padx=10, anchor="w")
        
        # Treeview for database content
        tree_frame = ttk.Frame(self.db_tab)
        tree_frame.pack(pady=5, padx=10, fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.db_tree = ttk.Treeview(tree_frame, columns=("Symbol", "Date", "Close", "Volume"),
                                   show="headings", yscrollcommand=scrollbar.set)
        self.db_tree.heading("Symbol", text="Symbol")
        self.db_tree.heading("Date", text="Date")
        self.db_tree.heading("Close", text="Close")
        self.db_tree.heading("Volume", text="Volume")
        
        self.db_tree.column("Symbol", width=100)
        self.db_tree.column("Date", width=120)
        self.db_tree.column("Close", width=100)
        self.db_tree.column("Volume", width=120)
        
        self.db_tree.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.db_tree.yview)
        
        ttk.Button(self.db_tab, text="🔄 Refresh Database View",
                  command=self.refresh_db_view).pack(pady=10)
        
    def generate_analytics(self):
        """Generate comprehensive analytics visualizations"""
        try:
            symbol = self._resolve_symbol(self.current_symbol.get())
        except Exception as e:
            messagebox.showwarning("Input Error", str(e))
            return
        
        self.update_results(f"📊 Generating analytics for {symbol}...\n")
        
        try:
            # Clear previous charts
            for widget in self.analytics_scroll_frame.winfo_children():
                widget.destroy()
            
            # Fetch data
            df = yf.download(symbol, period="1y", progress=False)
            df = _flatten_yf_columns(df)
            
            if df.empty:
                messagebox.showerror("Error", f"No data found for {symbol}")
                return
            
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Calculate indicators
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_50'] = df['Close'].rolling(50).mean()
            df['Daily_Return'] = df['Close'].pct_change() * 100
            df['Volatility'] = df['Daily_Return'].rolling(20).std()
            df['Month'] = df['Date'].dt.to_period('M')
            
            # 1. Price Trend Chart
            fig1 = Figure(figsize=(8, 3.5), dpi=90)
            fig1.patch.set_facecolor(COLOR_BG_MEDIUM)
            ax1 = fig1.add_subplot(111)
            ax1.set_facecolor(COLOR_BG_MEDIUM)
            ax1.plot(df['Date'], df['Close'], color=COLOR_ACCENT_TEAL, linewidth=2, label='Actual Price')
            ax1.set_title(f'{symbol} - Stock Price Trend Over Time', color=COLOR_TEXT_LIGHT, fontsize=11)
            ax1.set_xlabel('Date', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax1.set_ylabel('Closing Price', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax1.tick_params(colors=COLOR_TEXT_LIGHT, labelsize=8)
            ax1.legend(facecolor=COLOR_BG_MEDIUM, labelcolor=COLOR_TEXT_LIGHT, fontsize=8)
            ax1.grid(True, alpha=0.3)
            fig1.tight_layout()
            
            canvas1 = FigureCanvasTkAgg(fig1, master=self.analytics_scroll_frame)
            canvas1.draw()
            canvas1.get_tk_widget().pack(pady=10, padx=10, fill="x")
            
            # 2. Moving Average Comparison
            fig2 = Figure(figsize=(8, 3.5), dpi=90)
            fig2.patch.set_facecolor(COLOR_BG_MEDIUM)
            ax2 = fig2.add_subplot(111)
            ax2.set_facecolor(COLOR_BG_MEDIUM)
            ax2.plot(df['Date'], df['Close'], color='gray', alpha=0.5, label='Actual Price')
            ax2.plot(df['Date'], df['MA_20'], color=COLOR_PREDICTION_GOLD, linewidth=2, label='20-Day MA')
            ax2.plot(df['Date'], df['MA_50'], color=COLOR_ACCENT_TEAL, linewidth=2, label='50-Day MA')
            ax2.set_title(f'{symbol} - Moving Average Comparison', color=COLOR_TEXT_LIGHT, fontsize=11)
            ax2.set_xlabel('Date', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax2.set_ylabel('Price', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax2.tick_params(colors=COLOR_TEXT_LIGHT, labelsize=8)
            ax2.legend(facecolor=COLOR_BG_MEDIUM, labelcolor=COLOR_TEXT_LIGHT, fontsize=8)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            
            canvas2 = FigureCanvasTkAgg(fig2, master=self.analytics_scroll_frame)
            canvas2.draw()
            canvas2.get_tk_widget().pack(pady=10, padx=10, fill="x")
            
            # 3. Daily Returns Distribution
            fig3 = Figure(figsize=(8, 3.5), dpi=90)
            fig3.patch.set_facecolor(COLOR_BG_MEDIUM)
            ax3 = fig3.add_subplot(111)
            ax3.set_facecolor(COLOR_BG_MEDIUM)
            ax3.hist(df['Daily_Return'].dropna(), bins=50, color=COLOR_ACCENT_TEAL, alpha=0.7, edgecolor='black')
            ax3.axvline(0, color=COLOR_PREDICTION_GOLD, linestyle='--', linewidth=2)
            ax3.set_title(f'{symbol} - Daily Returns Distribution (%)', color=COLOR_TEXT_LIGHT, fontsize=11)
            ax3.set_xlabel('Daily Return (%)', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax3.set_ylabel('Frequency', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax3.tick_params(colors=COLOR_TEXT_LIGHT, labelsize=8)
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            
            canvas3 = FigureCanvasTkAgg(fig3, master=self.analytics_scroll_frame)
            canvas3.draw()
            canvas3.get_tk_widget().pack(pady=10, padx=10, fill="x")
            
            # 4. Volatility Over Time
            fig4 = Figure(figsize=(8, 3.5), dpi=90)
            fig4.patch.set_facecolor(COLOR_BG_MEDIUM)
            ax4 = fig4.add_subplot(111)
            ax4.set_facecolor(COLOR_BG_MEDIUM)
            ax4.plot(df['Date'], df['Volatility'], color=COLOR_PREDICTION_GOLD, linewidth=2, label='Volatility (20-day std)')
            ax4.set_title(f'{symbol} - Volatility Over Time', color=COLOR_TEXT_LIGHT, fontsize=11)
            ax4.set_xlabel('Date', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax4.set_ylabel('Volatility (%)', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax4.tick_params(colors=COLOR_TEXT_LIGHT, labelsize=8)
            ax4.legend(facecolor=COLOR_BG_MEDIUM, labelcolor=COLOR_TEXT_LIGHT, fontsize=8)
            ax4.grid(True, alpha=0.3)
            fig4.tight_layout()
            
            canvas4 = FigureCanvasTkAgg(fig4, master=self.analytics_scroll_frame)
            canvas4.draw()
            canvas4.get_tk_widget().pack(pady=10, padx=10, fill="x")
            
            # 5. Monthly Average Bar Chart
            monthly_avg = df.groupby('Month')['Close'].mean().reset_index()
            monthly_avg = monthly_avg.tail(6)
            
            fig5 = Figure(figsize=(8, 3.5), dpi=90)
            fig5.patch.set_facecolor(COLOR_BG_MEDIUM)
            ax5 = fig5.add_subplot(111)
            ax5.set_facecolor(COLOR_BG_MEDIUM)
            ax5.bar(range(len(monthly_avg)), monthly_avg['Close'], color=COLOR_ACCENT_TEAL, edgecolor='black')
            ax5.set_xticks(range(len(monthly_avg)))
            ax5.set_xticklabels(monthly_avg['Month'].astype(str), rotation=45, fontsize=8)
            ax5.set_title(f'{symbol} - Average Monthly Closing Price (Last 6 Months)', color=COLOR_TEXT_LIGHT, fontsize=11)
            ax5.set_xlabel('Month', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax5.set_ylabel('Average Closing Price', color=COLOR_TEXT_LIGHT, fontsize=9)
            ax5.tick_params(colors=COLOR_TEXT_LIGHT, labelsize=8)
            ax5.grid(True, alpha=0.3, axis='y')
            fig5.tight_layout()
            
            canvas5 = FigureCanvasTkAgg(fig5, master=self.analytics_scroll_frame)
            canvas5.draw()
            canvas5.get_tk_widget().pack(pady=10, padx=10, fill="x")
            
            self.update_results(f"✅ Generated 5 analytics charts for {symbol}", append=True)
            
        except Exception as e:
            messagebox.showerror("Analytics Error", str(e))
            self.update_results(f"❌ Error: {str(e)}", append=True)
    
    def plot_initial_chart(self):
        """Plot initial empty chart"""
        self.ax_price.clear()
        self.ax_price.text(0.5, 0.5, "Select a symbol and train/predict to view chart",
                          ha='center', va='center', color=COLOR_TEXT_LIGHT,
                          transform=self.ax_price.transAxes)
        self.ax_price.set_facecolor(COLOR_BG_MEDIUM)
        self.canvas_price.draw()
        
    def update_results(self, text, append=False):
        """Update results text widget"""
        if not append:
            self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.see(tk.END)
        self.root.update()
        
    def train_models(self):
        """Train ML models"""
        user_input = self.current_symbol.get()
        try:
            symbol = self._resolve_symbol(user_input)
        except Exception as e:
            messagebox.showwarning("Input Error", str(e))
            return
        
        self.update_results(f"🚀 Training models for {symbol}...\n")
        
        try:
            # Fetch data
            self.update_results("📥 Fetching data from Yahoo Finance...", append=True)
            df = yf.download(symbol, period="2y", progress=False)
            if df.empty:
                messagebox.showerror("Error", f"No data found for {symbol}")
                return

            # Normalize columns from yfinance (avoid MultiIndex/tuple columns)
            df = _flatten_yf_columns(df)
            
            # Store in database
            df_db = _flatten_yf_columns(df)
            df_reset = df_db.reset_index()
            df_to_db(df_reset, symbol)
            self.update_results(f"💾 Saved {len(df)} records to database", append=True)
            
            # Preprocess
            self.update_results("⚙️ Preprocessing data...", append=True)
            df_processed = self.preprocess_data(df)
            
            # Train models
            self.update_results("🎓 Training ML models...", append=True)
            results = self.train_ml_models(df_processed, symbol)
            
            self.update_results(f"\n✅ Training Complete!\n", append=True)
            self.update_results(f"Model Metrics:", append=True)
            for name, metrics in results.items():
                self.update_results(f"  {name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}",
                                  append=True)
            
            self.models_loaded = True
            self.plot_training_data(df, symbol)
            
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.update_results(f"❌ Error: {str(e)}", append=True)
            
    def predict_next_day(self):
        """Predict next day price"""
        try:
            symbol = self._resolve_symbol(self.current_symbol.get())
        except Exception as e:
            messagebox.showwarning("Input Error", str(e))
            return
        
        if not os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_LinearRegression.joblib")):
            messagebox.showinfo("Info", "Models not trained yet. Training now...")
            self.train_models()
            return
            
        self.update_results(f"🔮 Predicting next day for {symbol}...\n")
        
        try:
            df = yf.download(symbol, period="1y", progress=False)
            df = _flatten_yf_columns(df)
            df_processed = self.preprocess_data(df)
            
            scaler = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib"))
            features = self.get_feature_names()
            X = df_processed[features].iloc[-1:].astype(float)
            X_scaled = scaler.transform(X)
            
            # Load models and predict
            models = {}
            # Always attempt to load available models; skip missing
            lr_path = os.path.join(MODEL_DIR, f"{symbol}_LinearRegression.joblib")
            rf_path = os.path.join(MODEL_DIR, f"{symbol}_RandomForest.joblib")
            xgb_path = os.path.join(MODEL_DIR, f"{symbol}_XGBoost.joblib")
            if os.path.exists(lr_path):
                models["Linear Regression"] = joblib.load(lr_path)
            if os.path.exists(rf_path):
                models["Random Forest"] = joblib.load(rf_path)
            if HAS_XGB and os.path.exists(xgb_path):
                models["XGBoost"] = joblib.load(xgb_path)
            
            if not models:
                raise RuntimeError("No trained models found. Please train models first.")
            
            current_price = float(df['Close'].iloc[-1])
            predictions = {}
            
            for name, model in models.items():
                pred = float(model.predict(X_scaled)[0])
                predictions[name] = pred
                change = ((pred - current_price) / current_price) * 100
                self.update_results(f"  {name}: ₹{pred:.2f} ({change:+.2f}%)", append=True)
            
            ensemble = np.mean(list(predictions.values()))
            ensemble_change = ((ensemble - current_price) / current_price) * 100
            
            self.update_results(f"\n💹 Current Price: ₹{current_price:.2f}", append=True)
            self.update_results(f"🎯 Ensemble Prediction: ₹{ensemble:.2f} ({ensemble_change:+.2f}%)\n",
                              append=True)
            messagebox.showinfo(
                "Next Day Prediction",
                (
                    f"Symbol: {symbol}\n\n"
                    f"Current Price: ₹{current_price:.2f}\n"
                    f"Ensemble Prediction: ₹{ensemble:.2f} ({ensemble_change:+.2f}%)"
                ),
            )
            
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.update_results(f"❌ Error: {str(e)}", append=True)
            
    def hybrid_forecast(self):
        """30-day hybrid forecast"""
        try:
            symbol = self._resolve_symbol(self.current_symbol.get())
        except Exception as e:
            messagebox.showwarning("Input Error", str(e))
            return
            
        if not os.path.exists(os.path.join(MODEL_DIR, f"{symbol}_LinearRegression.joblib")):
            messagebox.showinfo("Info", "Models not trained yet. Training now...")
            self.train_models()
            return
            
        self.update_results(f"📊 Running 30-day forecast for {symbol}...\n")
        
        try:
            df = yf.download(symbol, period="1y", progress=False)
            df = _flatten_yf_columns(df)
            df_processed = self.preprocess_data(df)
            
            # Simple forecast (can be enhanced with ARIMA)
            scaler = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib"))
            model_lr = joblib.load(os.path.join(MODEL_DIR, f"{symbol}_LinearRegression.joblib"))
            
            current_price = float(df['Close'].iloc[-1])
            forecasts = []
            temp_df = df_processed.copy()
            
            for day in range(30):
                features = self.get_feature_names()
                X = temp_df[features].iloc[-1:].astype(float)
                X_scaled = scaler.transform(X)
                pred = float(model_lr.predict(X_scaled)[0])
                forecasts.append(pred)
                
                # Update temp_df with synthetic next-day row preserving feature columns
                new_row = temp_df.iloc[-1].copy()
                new_row['Close'] = pred
                # If OHLC not present or for synthetic step, align them to predicted close
                for c in ('Open', 'High', 'Low'):
                    if c in new_row:
                        new_row[c] = pred
                if 'Volume' in new_row and pd.isna(new_row['Volume']):
                    new_row['Volume'] = 0
                
                temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
                # Recompute indicators on the full dataframe (do not drop columns)
                temp_df = self.preprocess_data(temp_df)
            
            final_price = forecasts[-1]
            total_change = ((final_price - current_price) / current_price) * 100
            
            self.update_results(f"📈 30-Day Forecast Results:", append=True)
            self.update_results(f"  Current: ₹{current_price:.2f}", append=True)
            self.update_results(f"  Predicted (Day 30): ₹{final_price:.2f}", append=True)
            self.update_results(f"  Expected Change: {total_change:+.2f}%\n", append=True)
            messagebox.showinfo(
                "30-Day Forecast",
                (
                    f"Symbol: {symbol}\n\n"
                    f"Current Price: ₹{current_price:.2f}\n"
                    f"Predicted Price (Day 30): ₹{final_price:.2f}\n"
                    f"Expected Change: {total_change:+.2f}%"
                ),
            )
            
            # Plot forecast
            self.plot_forecast(df, forecasts, symbol)
            
        except Exception as e:
            messagebox.showerror("Forecast Error", str(e))
            self.update_results(f"❌ Error: {str(e)}", append=True)
            
    def import_csv_to_db(self):
        """Import CSV file to database"""
        filepath = filedialog.askopenfilename(title="Select CSV file",
                                             filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return
            
        try:
            df = pd.read_csv(filepath)
            symbol = self.current_symbol.get().strip().upper()
            if not symbol:
                symbol = os.path.splitext(os.path.basename(filepath))[0].upper()
                
            count = df_to_db(df, symbol)
            self.update_results(f"✅ Imported {count} records for {symbol}")
            messagebox.showinfo("Success", f"Imported {count} records for {symbol}")
            self.refresh_db_view()
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
            
    def export_symbol_csv(self):
        """Export symbol data to CSV"""
        symbol = self.current_symbol.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol")
            return
            
        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                               filetypes=[("CSV files", "*.csv")],
                                               initialfile=f"{symbol}.csv")
        if not filepath:
            return
            
        try:
            export_symbol_to_csv(symbol, filepath)
            self.update_results(f"✅ Exported {symbol} to {filepath}")
            messagebox.showinfo("Success", f"Exported {symbol} to CSV")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            
    def delete_symbol_from_db(self):
        """Delete symbol from database"""
        symbol = self.current_symbol.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Input Error", "Please enter a stock symbol")
            return
            
        confirmed = messagebox.askyesno("Confirm", 
                                        f"Delete all records for {symbol}?")
        if not confirmed:
            return
            
        try:
            count = delete_symbol(symbol)
            self.update_results(f"🗑️ Deleted {count} records for {symbol}")
            messagebox.showinfo("Success", f"Deleted {count} records")
            self.refresh_db_view()
        except Exception as e:
            messagebox.showerror("Delete Error", str(e))
            
    def list_all_symbols(self):
        """List all symbols in database with dialog"""
        try:
            symbols = list_symbols()
            if not symbols:
                messagebox.showinfo("Database", "No symbols found in database.")
                return
            
            # Create dialog to display symbols
            dialog = tk.Toplevel(self.root)
            dialog.title("Database Symbols")
            dialog.geometry("350x450")
            dialog.configure(bg=COLOR_BG_DARK)
            
            ttk.Label(dialog, text=f"Symbols in Database ({len(symbols)})",
                     style="Section.TLabel", background=COLOR_BG_DARK).pack(pady=10)
            
            # Listbox with scrollbar
            frame = tk.Frame(dialog, bg=COLOR_BG_DARK)
            frame.pack(pady=10, padx=20, fill="both", expand=True)
            
            scrollbar = tk.Scrollbar(frame)
            scrollbar.pack(side="right", fill="y")
            
            listbox = tk.Listbox(frame, bg=COLOR_BG_MEDIUM, fg=COLOR_TEXT_LIGHT,
                                font=("Inter", 11), yscrollcommand=scrollbar.set)
            listbox.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=listbox.yview)
            
            for symbol in symbols:
                listbox.insert(tk.END, f"  • {symbol}")
            
            def select_and_load():
                selection = listbox.curselection()
                if selection:
                    symbol_text = listbox.get(selection[0])
                    symbol = symbol_text.replace("  • ", "").strip()
                    self.current_symbol.set(symbol)
                    dialog.destroy()
                    self.update_results(f"Selected symbol: {symbol}")
            
            btn_frame = tk.Frame(dialog, bg=COLOR_BG_DARK)
            btn_frame.pack(pady=10)
            
            ttk.Button(btn_frame, text="Load Selected", command=select_and_load).pack(side="left", padx=5)
            ttk.Button(btn_frame, text="Close", command=dialog.destroy).pack(side="left", padx=5)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    # (Removed duplicate create_widgets definition to avoid packing conflicts)
            pass
            
    # Helper methods for ML
    def preprocess_data(self, df):
        """Preprocess stock data with technical indicators"""
        df = _flatten_yf_columns(df.copy())

        # Drop stray index columns from prior reset_index calls and normalize index
        for col in ("level_0", "index"):
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.reset_index(drop=True)
        
        # Basic sanity check
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            # Try a graceful fallback: derive OHLC from Close and set Volume=0
            if "Close" not in df.columns:
                raise ValueError(f"Missing required columns: {missing}. Got columns: {list(df.columns)}")
            if "Open" in missing:
                df["Open"] = df["Close"]
            if "High" in missing:
                df["High"] = df["Close"]
            if "Low" in missing:
                df["Low"] = df["Close"]
            if "Volume" in missing:
                df["Volume"] = 0.0
        
        df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
        
        # RSI
        delta = df["Close"].diff().to_numpy().flatten()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_series = pd.Series(gain, index=df.index)
        loss_series = pd.Series(loss, index=df.index)
        roll_up = gain_series.rolling(14, min_periods=1).mean()
        roll_down = loss_series.rolling(14, min_periods=1).mean()
        rs = roll_up / (roll_down + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        
        # Bollinger Bands
        df["STD20"] = df["Close"].rolling(20, min_periods=1).std()
        df["BB_upper"] = df["MA20"] + (2 * df["STD20"])
        df["BB_lower"] = df["MA20"] - (2 * df["STD20"])
        
        # Lags and returns
        df["Close_lag1"] = df["Close"].shift(1)
        df["Close_lag3"] = df["Close"].shift(3)
        df["Return_1d"] = df["Close"].pct_change(1)
        df["Return_5d"] = df["Close"].pct_change(5)
        df["TimeIndex"] = np.arange(len(df))
        
        df["Target"] = df["Close"].shift(-1)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method="bfill", inplace=True)
        df.fillna(method="ffill", inplace=True)
        
        return df
        
    def get_feature_names(self):
        """Get list of feature names"""
        return ["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI",
                "MACD", "BB_upper", "BB_lower", "Close_lag1", "Close_lag3",
                "Return_1d", "Return_5d", "TimeIndex"]
        
    def train_ml_models(self, df, symbol):
        """Train ML models"""
        features = self.get_feature_names()
        X = df[features].astype(float)
        y = df["Target"].astype(float)
        
        # Remove last row if target is NaN
        if pd.isna(y.iloc[-1]):
            X = X.iloc[:-1]
            y = y.iloc[:-1]
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        split = int(TRAIN_TEST_SPLIT * len(X))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.05,
                                              random_state=RANDOM_STATE, verbosity=0)
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            r2 = float(1 - (np.sum((y_test - y_pred) ** 2) / 
                           np.sum((y_test - y_test.mean()) ** 2)))
            results[name] = {"rmse": rmse, "r2": r2}
            
            joblib.dump(model, os.path.join(MODEL_DIR, f"{symbol}_{name}.joblib"))
        
        joblib.dump(scaler, os.path.join(MODEL_DIR, f"{symbol}_scaler.joblib"))
        
        return results
        
    def plot_training_data(self, df, symbol):
        """Plot training data"""
        self.ax_price.clear()
        self.ax_price.plot(df.index[-200:], df['Close'].tail(200), 
                          color=COLOR_ACCENT_TEAL, linewidth=2, label='Close Price')
        self.ax_price.set_title(f"{symbol} - Training Data", 
                               color=COLOR_TEXT_LIGHT, fontsize=14)
        self.ax_price.set_xlabel("Date", color=COLOR_TEXT_LIGHT)
        self.ax_price.set_ylabel("Price", color=COLOR_TEXT_LIGHT)
        self.ax_price.tick_params(colors=COLOR_TEXT_LIGHT)
        self.ax_price.legend(facecolor=COLOR_BG_DARK, labelcolor=COLOR_TEXT_LIGHT)
        self.ax_price.grid(True, alpha=0.3)
        self.fig_price.tight_layout()
        self.canvas_price.draw()
        
    def plot_db_data(self, df, symbol):
        """Plot data from database"""
        self.ax_price.clear()
        self.ax_price.plot(pd.to_datetime(df['date']), df['close'], 
                          color=COLOR_ACCENT_TEAL, linewidth=2, label='Close Price')
        self.ax_price.set_title(f"{symbol} - Database Data", 
                               color=COLOR_TEXT_LIGHT, fontsize=14)
        self.ax_price.set_xlabel("Date", color=COLOR_TEXT_LIGHT)
        self.ax_price.set_ylabel("Price", color=COLOR_TEXT_LIGHT)
        self.ax_price.tick_params(colors=COLOR_TEXT_LIGHT)
        self.ax_price.legend(facecolor=COLOR_BG_DARK, labelcolor=COLOR_TEXT_LIGHT)
        self.ax_price.grid(True, alpha=0.3)
        self.fig_price.tight_layout()
        self.canvas_price.draw()
        
    def plot_forecast(self, df, forecasts, symbol):
        """Plot forecast"""
        self.ax_price.clear()
        
        # Plot historical
        historical = df['Close'].tail(60)
        self.ax_price.plot(range(len(historical)), historical.values, 
                          color=COLOR_ACCENT_TEAL, linewidth=2, label='Historical')
        
        # Plot forecast
        forecast_x = range(len(historical), len(historical) + len(forecasts))
        self.ax_price.plot(forecast_x, forecasts, 
                          color=COLOR_PREDICTION_GOLD, linewidth=2, 
                          linestyle='--', label='30-Day Forecast')
        
        self.ax_price.set_title(f"{symbol} - 30-Day Forecast", 
                               color=COLOR_TEXT_LIGHT, fontsize=14)
        self.ax_price.set_xlabel("Days", color=COLOR_TEXT_LIGHT)
        self.ax_price.set_ylabel("Price", color=COLOR_TEXT_LIGHT)
        self.ax_price.tick_params(colors=COLOR_TEXT_LIGHT)
        self.ax_price.legend(facecolor=COLOR_BG_DARK, labelcolor=COLOR_TEXT_LIGHT)
        self.ax_price.grid(True, alpha=0.3)
        self.fig_price.tight_layout()
        self.canvas_price.draw()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
