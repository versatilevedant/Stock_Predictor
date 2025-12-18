# 📈 Stock Market Predictor - Integrated System

A comprehensive stock market prediction system combining Machine Learning, Database Management, and Interactive Visualizations.

## 🌟 Features

### 🤖 Machine Learning
- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost
- **Time Series Analysis**: ARIMA integration for temporal patterns
- **Ensemble Predictions**: Weighted combination of models
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Next-Day Predictions**: High-accuracy short-term forecasts
- **30-Day Forecasts**: Long-term trend predictions

### 🗄️ Database Management
- **SQLite Integration**: Persistent stock data storage
- **CSV Import/Export**: Easy data management
- **Multi-Symbol Support**: Track multiple stocks
- **Data Deduplication**: Automatic handling of duplicate records
- **Query & Filter**: Retrieve specific date ranges

### 📊 Visualization
- **Interactive Charts**: Real-time price trend visualization
- **Technical Analysis**: Moving averages, volatility, returns
- **Multi-Chart Display**: Historical, predicted, and forecast views
- **Customizable Timeframes**: View data across different periods

### 🖥️ User Interface
- **Modern GUI**: Professional dark-themed interface
- **Tabbed Layout**: Organized workspace (Price Chart, Analytics, Database)
- **Real-time Updates**: Live feedback on operations
- **Multi-Action Support**: Train, predict, analyze, export in one place

## 📂 Project Structure

```
proj/
├── config.py                          # Central configuration
├── database.py                        # Database operations
├── file_manager.py                    # CSV import/export utilities
├── stoc_market.py                     # Standalone visualization module
├── Stock Predictor Final.py           # Advanced ML predictor (CLI)
├── frontend                           # Simple GUI frontend
├── main_db_integration_example.py     # Database + GUI demo
├── integrated_app.py                  # 🌟 MAIN APPLICATION (complete integration)
├── test_imports.py                    # Dependency checker
├── requirements.txt                   # Python dependencies
├── stock_data.db                      # SQLite database
├── data/                              # Additional data storage
├── models/                            # Trained ML models
└── assets/                            # GUI assets
```

## 🚀 Quick Start

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python test_imports.py
```

### Running the Application

#### Option 1: Integrated Application (Recommended)
The complete all-in-one solution:
```bash
python integrated_app.py
```

**Features:**
- Full ML training and prediction
- Database management
- Visualization dashboards
- CSV import/export

#### Option 2: Advanced ML Predictor (CLI)
For detailed quantitative analysis:
```bash
python "Stock Predictor Final.py"
```

**Provides:**
- Next-day price prediction
- 30-day hybrid forecast
- Probability of upward movement
- Model performance metrics
- Feature importance analysis

#### Option 3: Simple Frontend
Basic prediction interface:
```bash
python frontend
```

#### Option 4: Database Demo
Database operations with GUI:
```bash
python main_db_integration_example.py
```

#### Option 5: Visualization Script
Generate comprehensive charts:
```bash
python stoc_market.py
```

## 📖 Usage Guide

### Training Models

1. Launch `integrated_app.py`
2. Enter a stock symbol (e.g., `RELIANCE.NS`, `AAPL`, `TSLA`)
3. Click **"🎓 Train Models"**
4. Wait for training to complete
5. View model metrics in the results panel

### Making Predictions

**Next-Day Prediction:**
- Click **"🔮 Predict Next Day"**
- View individual model predictions and ensemble result

**30-Day Forecast:**
- Click **"📊 30-Day Forecast"**
- View extended prediction with trend analysis

### Database Operations

**Import Data:**
- Click **"📂 Import CSV to DB"**
- Select your CSV file (must contain Date, Close, Volume columns)

**Export Data:**
- Click **"💾 Export Symbol to CSV"**
- Choose save location

**View Database:**
- Switch to **"🗄️ Database"** tab
- Click **"🔄 Refresh Database View"**

**List All Symbols:**
- Click **"📋 List All Symbols"**
- View all stocks in database

**Load from Database:**
- Enter symbol
- Click **"📥"** button (next to symbol entry)

### Stock Symbols Format

- **Indian Stocks**: Add `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`)
- **US Stocks**: Use ticker directly (e.g., `AAPL`, `TSLA`, `GOOGL`)
- **Other Markets**: Use appropriate Yahoo Finance suffix

## 🔧 Configuration

Edit `config.py` to customize:

- **Database path**: Change `DB_PATH`
- **Model directory**: Change `MODEL_DIR`
- **Default symbol**: Change `DEFAULT_SYMBOL`
- **Colors**: Modify GUI theme colors
- **Model parameters**: Adjust training settings

## 📊 Technical Indicators Explained

| Indicator | Purpose |
|-----------|------|
| **MA20/MA50** | 20-day and 50-day Moving Averages (trend) |
| **RSI** | Relative Strength Index (momentum, 0-100) |
| **MACD** | Moving Average Convergence Divergence (trend changes) |
| **Bollinger Bands** | Volatility and price levels |
| **Daily Returns** | Day-to-day percentage changes |
| **Volatility** | Price fluctuation measure |

## 🧠 Machine Learning Models

### Linear Regression
- Fast, interpretable baseline
- Good for linear trends
- Weight: 0.6-0.78 in ensemble

### Random Forest
- Captures non-linear patterns
- Robust to outliers
- Weight: 0.11-0.2 in ensemble

### XGBoost
- High accuracy for complex patterns
- Gradient boosting technique
- Weight: 0.11-0.2 in ensemble

### ARIMA (Optional)
- Time series temporal smoothing
- Used in 30-day hybrid forecasts
- Blended with ML predictions

## 📁 Database Schema

```sql
CREATE TABLE stock_history (
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
);
```

## 🐛 Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### tkinter Not Found (Linux)
```bash
sudo apt-get install python3-tk
```

### No Data for Symbol
- Verify symbol format (add `.NS` for Indian stocks)
- Check internet connection
- Try a different symbol

### Model Not Found
- Train models first using **"🎓 Train Models"**
- Models are saved per symbol

### Database Locked
- Close other instances of the application
- Delete `stock_data.db-journal` if it exists

## 📈 Example Workflows

### Workflow 1: Analyze a New Stock
1. Enter symbol (e.g., `AAPL`)
2. Click **Train Models** (fetches data + trains)
3. View training chart
4. Click **Predict Next Day**
5. Click **30-Day Forecast**
6. Export results to CSV

### Workflow 2: Import Historical Data
1. Prepare CSV with columns: Date, Open, High, Low, Close, Volume
2. Click **Import CSV to DB**
3. Select file
4. View in Database tab
5. Train models on imported data

### Workflow 3: Compare Multiple Stocks
1. Train models for stock A
2. Note predictions
3. Change symbol to stock B
4. Train models
5. Compare results in Results panel

## 🎯 Advanced Features

### Model Persistence
- Trained models saved to `models/` directory
- Format: `{SYMBOL}_{MODEL_NAME}.joblib`
- Scalers saved separately: `{SYMBOL}_scaler.joblib`

### Date Range Queries
```python
from database import load_symbol
df = load_symbol("AAPL", start_date="2023-01-01", end_date="2023-12-31")
```

### Custom Visualizations
```python
from stoc_market import visualize_stock_trends
visualize_stock_trends("RELIANCE.NS", period="6mo")
```

## 🤝 Contributing

This is a mini-project for educational purposes. Feel free to:
- Add new ML models
- Improve UI/UX
- Add more technical indicators
- Enhance forecasting methods

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- Not financial advice
- Past performance ≠ future results
- Always do your own research before investing
- Consult professional financial advisors for investment decisions

## 📜 License

This project is open-source and available for educational use.

## 👨‍💻 Components

- **ML Models**: scikit-learn, XGBoost, statsmodels
- **Data Source**: Yahoo Finance (yfinance)
- **Database**: SQLite3
- **GUI**: Tkinter
- **Visualization**: Matplotlib, Seaborn

---

**Happy Predicting! 📈🚀**
