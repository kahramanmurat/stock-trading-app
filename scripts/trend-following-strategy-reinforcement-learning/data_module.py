# data_module.py
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

def fetch_data(ticker_symbol, start_date="2000-01-01"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker_symbol, start=start_date, end=end_date)
    return df

def preprocess_data(df):
    df['FastSMA']  = df['Close'].rolling(16).mean()
    df['SlowSMA']  = df['Close'].rolling(33).mean()
    df['LogReturn'] = np.log(df['Close']).diff()
    return df
