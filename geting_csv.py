import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Ticker list and date range
tickers = ["META", "NVDA", "BLK", "AMZN", "BSX", "UBER", "WMT", "COP", "NEE", "STLD"]
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)

# Download historical close prices
data = yf.download(tickers, start=start_date, end=end_date, progress=False)
close = data['Close'].dropna()

# Calculate daily log returns
log_returns = np.log(close / close.shift(1)).dropna()

# Save log returns to CSV
log_returns.to_csv("log_returnsss.csv")

print("Log returns successfully saved to 'log_returns.csv'")
