# 🚀 Quick Start Guide

## ⚡ Fastest Way to Get Started

```bash
# 1. Install dependencies (one time)
pip install -r requirements.txt

# 2. Run the main app
python integrated_app.py
```

## 🎯 First Time Usage

1. **Launch the app**: `python integrated_app.py`
2. **Enter a symbol**: Type `AAPL` or `RELIANCE.NS` in the text box
3. **Click "🎓 Train Models"**: Wait 1-2 minutes while it:
   - Downloads stock data
   - Trains ML models
   - Saves to database
4. **Click "🔮 Predict Next Day"**: See tomorrow's predicted price
5. **Click "📊 30-Day Forecast"**: See long-term trend

**That's it!** 🎉

## 📋 Cheat Sheet

### Common Stock Symbols
| Market | Symbol Format | Examples |
|--------|--------------|----------|
| US Stocks | Ticker only | `AAPL`, `TSLA`, `GOOGL`, `MSFT` |
| Indian Stocks | Ticker + .NS | `RELIANCE.NS`, `TCS.NS`, `INFY.NS` |
| Other Markets | Ticker + suffix | Check Yahoo Finance |

### Button Quick Reference

| Button | What It Does | When to Use |
|--------|-------------|-------------|
| 🎓 Train Models | Download data + train ML | First time or new symbol |
| 🔮 Predict Next Day | Tomorrow's price | After training |
| 📊 30-Day Forecast | Month ahead prediction | After training |
| 📂 Import CSV | Load CSV → Database | Have historical data file |
| 💾 Export Symbol | Save DB → CSV | Want to backup data |
| 📥 (next to symbol) | Load from DB | View existing data |
| 📋 List All Symbols | Show DB contents | See what's stored |
| 🗑️ Delete Symbol | Remove from DB | Clean up database |

### Tabs
- **📈 Price Chart**: Main visualization
- **📊 Analytics**: Technical indicators (coming soon)
- **🗄️ Database**: View stored data

## 🎬 Example Session

```bash
$ python integrated_app.py

# In the app:
1. Type: AAPL
2. Click: 🎓 Train Models
   → Downloads ~2 years of Apple stock data
   → Trains 3 ML models
   → Shows metrics in Results panel
   
3. Click: 🔮 Predict Next Day
   → Current Price: $178.50
   → Predicted: $179.20 (+0.39%)
   
4. Click: 📊 30-Day Forecast
   → Shows 30-day price path
   → Final predicted price: $185.00
   
5. Click: 💾 Export Symbol to CSV
   → Saves AAPL.csv with all data
```

## 🐛 Quick Troubleshooting

### Problem: Import errors
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Problem: "No module named 'tkinter'"
**Solution (Linux):**
```bash
sudo apt-get install python3-tk
```

### Problem: No data for symbol
**Solutions:**
- Try `AAPL` first (guaranteed to work)
- For Indian stocks, add `.NS`: `RELIANCE.NS`
- Check internet connection

### Problem: Models not found
**Solution:**
- Click "🎓 Train Models" first
- Models are saved per symbol
- Each stock needs its own training

## 📁 File Overview

| File | Purpose |
|------|---------|
| `integrated_app.py` | **Main app** - Use this! |
| `run.py` | Menu launcher (optional) |
| `config.py` | Settings (edit to customize) |
| `database.py` | DB functions (don't edit) |
| `requirements.txt` | Dependencies list |
| `README.md` | Full documentation |

## 💡 Pro Tips

1. **Train once, predict many**: After training, you can predict multiple times
2. **Database persistence**: Trained models and data saved between sessions
3. **Compare stocks**: Train different symbols, compare predictions
4. **Export everything**: Use CSV export to analyze in Excel/Google Sheets
5. **Check results panel**: All important info appears in the left panel

## 🎓 Learning Path

### Beginner
1. Run `python integrated_app.py`
2. Try with `AAPL`
3. Train → Predict → Explore

### Intermediate
1. Import your own CSV data
2. Try Indian stocks (`.NS`)
3. Use database view tab
4. Export and analyze results

### Advanced
1. Edit `config.py` to customize
2. Use `Stock Predictor Final.py` for CLI
3. Run `stoc_market.py` for pure visualization
4. Modify `integrated_app.py` to add features

## 🆘 Need More Help?

1. **Full guide**: Read `README.md`
2. **Integration details**: Check `INTEGRATION_SUMMARY.md`
3. **Test setup**: Run `python test_imports.py`
4. **All options**: Run `python run.py`

## ⚡ Command Reference

```bash
# Main application (RECOMMENDED)
python integrated_app.py

# Alternative launchers
python run.py                          # Menu interface
python "Stock Predictor Final.py"      # Advanced CLI
python frontend                        # Simple GUI
python stoc_market.py                  # Visualizations only

# Utilities
python test_imports.py                 # Check dependencies
pip install -r requirements.txt        # Install packages
```

## 🎯 Success Checklist

- [ ] Installed dependencies
- [ ] Ran test_imports.py (all ✅)
- [ ] Launched integrated_app.py
- [ ] Trained a model (e.g., AAPL)
- [ ] Made a prediction
- [ ] Exported data to CSV
- [ ] Viewed database tab

**All checked? You're ready to go! 🚀**

---

**Remember**: Start with `python integrated_app.py` and experiment!
