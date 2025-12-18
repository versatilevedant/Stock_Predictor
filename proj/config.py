"""
config.py
---------
Central configuration for the Stock Market Predictor application.
"""
import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Database
DB_PATH = os.path.join(BASE_DIR, "stock_data.db")

# Model parameters
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# Default stock symbol (leave as a common example; you can type any ticker)
DEFAULT_SYMBOL = "AAPL"

# GUI Colors
COLOR_BG_DARK = "#0A192F"
COLOR_BG_MEDIUM = "#112240"
COLOR_ACCENT_TEAL = "#64FFDA"
COLOR_TEXT_LIGHT = "#CCD6F6"
COLOR_TEXT_DARK = "#0A192F"
COLOR_PREDICTION_GOLD = "#FFD700"

# Create necessary directories
for directory in [DATA_DIR, MODEL_DIR, ASSETS_DIR]:
    os.makedirs(directory, exist_ok=True)
