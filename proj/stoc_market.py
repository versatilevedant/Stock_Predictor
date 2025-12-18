import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import yfinance as yf

sns.set(style="whitegrid")

def visualize_stock_trends(symbol="RELIANCE.NS", period="1y"):
    """
    Generate comprehensive stock market visualizations.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS', 'AAPL')
        period: Time period ('1y', '6mo', '3mo', etc.)
    """
    # Fetch data
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    present_date = datetime.now()
    past = df[df['Date'] < present_date]
    present = df[df['Date'] == present_date]
    future = df[df['Date'] > present_date]

    # Historical Price Trend Line
    plt.figure(figsize=(10,5))
    plt.plot(past['Date'], past['Close'], color='red', label='Past')
    if not present.empty:
        plt.scatter(present['Date'], present['Close'], color='green', s=80, label='Present')
    if 'Predicted_Close' in df.columns and not df[df['Date'] > present_date].empty:
        combined_dates = pd.concat([present['Date'], future['Date']]) if not present.empty else future['Date']
        combined_values = pd.concat([present['Close'], future['Predicted_Close']]) if not present.empty else future['Predicted_Close']
        plt.plot(combined_dates, combined_values, color='blue', label='Predicted Future')
    plt.title(f'{symbol} - Stock Price Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Moving Average Comparison
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Close'], color='gray', label='Actual Price')
    plt.plot(df['Date'], df['MA_20'], color='red', label='20-Day MA')
    plt.plot(df['Date'], df['MA_50'], color='blue', label='50-Day MA')
    if not present.empty:
        plt.scatter(present['Date'], present['Close'], color='green', s=80, label='Present')
    plt.title(f'{symbol} - Moving Average Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Daily Returns Distribution
    df['Daily_Return'] = df['Close'].pct_change() * 100
    plt.figure(figsize=(8,5))
    sns.histplot(df['Daily_Return'].dropna(), kde=True, color='red')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'{symbol} - Daily Returns Distribution (%)')
    plt.xlabel('Daily Return (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


    # Volatility Over Time
    df['Volatility'] = df['Daily_Return'].rolling(20).std()
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Volatility'], color='red', label='Volatility (20-day std)')
    plt.axvline(present_date, color='green', linestyle='--', label='Present')
    plt.title(f'{symbol} - Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Monthly Average Bar Chart 
    df['Month'] = df['Date'].dt.to_period('M')  
    monthly_avg = df.groupby('Month')['Close'].mean().reset_index()
    monthly_avg = monthly_avg.tail(6)
    plt.figure(figsize=(10,5))
    sns.barplot(x=monthly_avg['Month'].astype(str), y=monthly_avg['Close'], color='red')
    plt.title(f'{symbol} - Average Monthly Closing Price (Last 6 Months)')
    plt.xlabel('Month')
    plt.ylabel('Average Closing Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., RELIANCE.NS, AAPL, TSLA): ").strip().upper()
    if not symbol:
        symbol = "RELIANCE.NS"
    print(f"\nGenerating visualizations for {symbol}...\n")
    try:
        visualize_stock_trends(symbol)
    except Exception as e:
        print(f"Error: {e}")


