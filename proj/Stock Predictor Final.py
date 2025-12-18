"""
Stock_Predictor_Final_Quant.py
--------------------------------
Quant-style stock predictor combining:
  - Linear Regression (baseline)
  - Random Forest (nonlinear feature discovery)
  - XGBoost (nonlinear alpha signal)
  - ARIMA (temporal smoothing)
Also builds a probability model for next-day direction.

Outputs:
  • Next-day ensemble price
  • Upward-movement probability
  • 30-day hybrid (ML + ARIMA) final price
  • Feature importances
"""

import warnings, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import joblib

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# Data Fetch
# ---------------------------------------------------
def fetch_data(symbol="RELIANCE.NS", start="2015-01-01"):
    end = datetime.utcnow().strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance>=0.2 returns a MultiIndex (e.g. ('Close','SYMB')); flatten to keep original names
        df.columns = [col[0] if col[0] else col[-1] for col in df.columns]
    df["Date_preserved"] = pd.to_datetime(df.index)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------
# Feature Engineering
# ---------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    # --- Moving averages ---
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()

    # --- RSI (14-day) ---
    delta = df["Close"].diff().to_numpy().flatten()          # ensure 1D
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)

    roll_up = gain_series.rolling(14, min_periods=1).mean()
    roll_down = loss_series.rolling(14, min_periods=1).mean()

    rs = roll_up / (roll_down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # --- Bollinger Bands ---
    df["STD20"] = df["Close"].rolling(20, min_periods=1).std()
    df["BB_upper"] = df["MA20"] + (2 * df["STD20"])
    df["BB_lower"] = df["MA20"] - (2 * df["STD20"])

    # --- Lags / Returns / Time Index ---
    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag3"] = df["Close"].shift(3)
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["TimeIndex"] = np.arange(len(df))

    # --- Target ---
    df["Target"] = df["Close"].shift(-1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    if pd.isna(df["Target"].iloc[-1]):
        df = df.iloc[:-1]

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------
# Model Training
# ---------------------------------------------------
def train_models(symbol="RELIANCE.NS"):
    df = preprocess_data(fetch_data(symbol))

    features = [
        "Open","High","Low","Close","Volume","MA20","MA50","RSI",
        "MACD","BB_upper","BB_lower","Close_lag1","Close_lag3",
        "Return_1d","Return_5d","TimeIndex"
    ]
    X = df[features].astype(float)
    y = df["Target"].astype(float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    split = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0)
    }

    results, importances = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        results[name] = {"rmse": rmse, "r2": r2}

        if hasattr(model, "feature_importances_"):
            imp = dict(sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]))
            importances[name] = imp
        else:
            importances[name] = {}

        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("✅ Model Training Complete")
    print("Metrics:", results)
    return df, features, scaler, results, importances


# ---------------------------------------------------
# Probability (Up/Down)
# ---------------------------------------------------
def train_probability_model(df):
    df = df.copy()
    df["Up"] = (df["Target"] > df["Close"]).astype(int)
    feats = ["Close","MA20","MA50","RSI","MACD","Close_lag1","Return_1d","Return_5d","TimeIndex"]
    X, y = df[feats], df["Up"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)
    joblib.dump(clf, os.path.join(MODEL_DIR, "ProbModel.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "ProbScaler.joblib"))
    acc = clf.score(X_scaled, y)
    print(f"🎯 Direction Classifier Accuracy: {acc*100:.2f}%")
    return clf, scaler


# ---------------------------------------------------
# Next-Day Prediction
# ---------------------------------------------------
def predict_next_day(symbol, features):
    df = preprocess_data(fetch_data(symbol))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    X_scaled = scaler.transform(df[features].iloc[-1:].astype(float))

    models = {
        "LinearRegression": joblib.load(os.path.join(MODEL_DIR, "LinearRegression.joblib")),
        "RandomForest": joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "XGBoost.joblib")),
    }

    preds = {name: float(m.predict(X_scaled)[0]) for name, m in models.items()}
    ensemble = np.average(list(preds.values()), weights=[0.78, 0.11, 0.11])

    # Direction probability
    prob_clf = joblib.load(os.path.join(MODEL_DIR, "ProbModel.joblib"))
    prob_scaler = joblib.load(os.path.join(MODEL_DIR, "ProbScaler.joblib"))
    prob_feats = ["Close","MA20","MA50","RSI","MACD","Close_lag1","Return_1d","Return_5d","TimeIndex"]
    prob_input = prob_scaler.transform(df[prob_feats].iloc[-1:].astype(float))
    up_prob_logistic = float(prob_clf.predict_proba(prob_input)[0][1])

    # Blend logistic probability with a smooth momentum-based adjustment to avoid coin-flip outputs
    current_price = float(df["Close"].iloc[-1])
    predicted_price = ensemble
    pred_return = (predicted_price - current_price) / current_price
    return_vol = df["Return_1d"].rolling(20).std().iloc[-1]

    if np.isnan(return_vol) or return_vol <= 1e-6:
        risk_adjust = 0.0
    else:
        risk_adjust = np.tanh(pred_return / (return_vol + 1e-9))

    price_prob = 0.5 + 0.25 * risk_adjust
    up_prob = np.clip(0.7 * up_prob_logistic + 0.3 * price_prob, 0.05, 0.95)

    return preds, ensemble, up_prob


# ---------------------------------------------------
# Hybrid Forecast (ML + ARIMA)
# ---------------------------------------------------
def hybrid_forecast(symbol, features, days=30):
    df = preprocess_data(fetch_data(symbol))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    model_lr = joblib.load(os.path.join(MODEL_DIR, "LinearRegression.joblib"))
    model_rf = joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib"))
    model_xgb = joblib.load(os.path.join(MODEL_DIR, "XGBoost.joblib"))

    arima = ARIMA(df["Close"], order=(5,1,0)).fit()
    arima_pred = arima.forecast(steps=days)
    if hasattr(arima_pred, "to_numpy"):
        arima_path = arima_pred.to_numpy().flatten()
    else:
        arima_path = np.array(arima_pred).flatten()

    temp = df.copy()
    preds_ml = []
    for idx in range(days):
        last = temp[features].iloc[-1:].astype(float)
        scaled = scaler.transform(last)
        p_lr = model_lr.predict(scaled)[0]
        p_rf = model_rf.predict(scaled)[0]
        p_xgb = model_xgb.predict(scaled)[0]
        hybrid_ml = 0.6*p_lr + 0.2*p_rf + 0.2*p_xgb

        arima_component = arima_path[idx] if idx < len(arima_path) else hybrid_ml
        combined_price = 0.7*hybrid_ml + 0.3*arima_component
        preds_ml.append(combined_price)
        new_row = temp.iloc[-1].copy()
        new_row["Close"] = combined_price
        temp.loc[len(temp)] = new_row
        temp = preprocess_data(temp)

    final_price = preds_ml[-1]
    return float(final_price)


# ---------------------------------------------------
# Main
# ---------------------------------------------------
if __name__ == "__main__":
    user_symbol = input("📊 Enter stock symbol (e.g., RELIANCE.NS, TCS.NS, AAPL, TSLA): ").strip().upper()
    symbol = user_symbol if user_symbol else "RELIANCE.NS"
    print(f"\n🚀 Running smart quant forecast for {symbol}...\n")


    df, features, scaler, results, importances = train_models(symbol)
    train_probability_model(df)

    preds, ensemble, up_prob = predict_next_day(symbol, features)
    current_price = float(df["Close"].iloc[-1])
    print(f"\n💹 Current Close Price: ₹{current_price:.2f}")
    print("\n📈 Next-Day Predictions:")
    for k,v in preds.items():
        print(f"  {k:<15}: ₹{v:.2f}")
    print(f"\n🔹 Ensemble (weighted LR+RF+XGB): ₹{ensemble:.2f}")
    print(f"🔹 Upward Movement Probability: {up_prob*100:.2f}%")

    final_30day = hybrid_forecast(symbol, features, days=30)
    print(f"\n🔮 30-Day Hybrid Forecast (final price): ₹{final_30day:.2f}")

    print("\n✅ Forecasting Complete. Models saved to /models.\n")
