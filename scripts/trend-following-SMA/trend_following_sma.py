import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os

def download_data(ticker, start_date="2000-01-01"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def plot_adjusted_close(df, ticker, output_folder):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['adj_close'], label='Adjusted Close Price')
    plt.title(f'{ticker} Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'adjusted_close_price.png'))
    plt.close()

def add_log_return(df):
    df['logreturn'] = np.log(df['close']).diff().shift(-1)
    return df

def add_sma(df, fast_period, slow_period):
    df['fastsma'] = df['close'].rolling(fast_period).mean()
    df['slowsma'] = df['close'].rolling(slow_period).mean()
    df['signal'] = np.where(df['fastsma'] >= df['slowsma'], 1, 0)
    df['prevsignal'] = df['signal'].shift(1)
    df['buy'] = (df['prevsignal'] == 0) & (df['signal'] == 1)
    df['sell'] = (df['prevsignal'] == 1) & (df['signal'] == 0)
    return df

def plot_sma(df, ticker, output_folder):
    df[['close', 'fastsma', 'slowsma']].iloc[-300:].plot(figsize=(10, 5))
    plt.title(f'{ticker} Close Price with SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig(os.path.join(output_folder, 'sma_plot.png'))
    plt.close()

def assign_is_invested(df):
    is_invested = False
    is_invested_list = []
    for index, row in df.iterrows():
        if is_invested and row['sell']:
            is_invested = False
        if not is_invested and row['buy']:
            is_invested = True
        is_invested_list.append(is_invested)
    return is_invested_list

def calculate_algo_log_return(df):
    df['algo_log_return'] = df['is_invested'] * df['logreturn']
    return df

def trend_following(df, fast, slow, Ntest=1000):
    df = add_sma(df, fast, slow)
    train = df.iloc[:-Ntest]
    test = df.iloc[-Ntest:]

    df.loc[df.index <= train.index[-1], 'is_invested'] = assign_is_invested(train)
    df.loc[df.index > train.index[-1], 'is_invested'] = assign_is_invested(test)

    df = calculate_algo_log_return(df)

    return df.loc[train.index, 'algo_log_return'][:-1].sum(), df.loc[test.index, 'algo_log_return'][:-1].sum()

def grid_search(df):
    best_fast = None
    best_slow = None
    best_score = float('-inf')
    for fast in range(3, 30):
        for slow in range(fast + 5, 50):
            score, _ = trend_following(df, fast, slow)
            if score > best_score:
                best_fast = fast
                best_slow = slow
                best_score = score
    return best_fast, best_slow, trend_following(df, best_fast, best_slow)

def calculate_statistics(df):
    train = df.iloc[:-1000]
    test = df.iloc[-1000:]
    
    train_sr = train['algo_log_return'].mean() / train['algo_log_return'].std()
    test_sr = test['algo_log_return'].mean() / test['algo_log_return'].std()
    
    train_buy_hold_sr = train['logreturn'].mean() / train['logreturn'].std()
    test_buy_hold_sr = test['logreturn'].mean() / test['logreturn'].std()
    
    return (train_sr, test_sr), (train_buy_hold_sr, test_buy_hold_sr)

def plot_wealth(df, output_folder):
    train = df.iloc[:-1000].copy()
    train['cum_log_return'] = train['algo_log_return'].cumsum().shift(1)
    train['cumwealth'] = train.iloc[0]['close'] * np.exp(train['cum_log_return'])
    train[['close', 'slowsma', 'fastsma', 'cumwealth']].plot(figsize=(20, 10))
    plt.title('Wealth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.savefig(os.path.join(output_folder, 'wealth_over_time.png'))
    plt.close()

if __name__ == "__main__":
    strategy_name = 'Trend_Following_SMA'
    output_folder = os.path.join('plot_data', strategy_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ticker = 'SPY'
    print(f"Downloading data for {ticker}...")
    df = download_data(ticker)
    print(f"Data downloaded. Plotting adjusted close prices...")
    plot_adjusted_close(df, ticker, output_folder)
    
    print("Adding log returns...")
    df = add_log_return(df)
    print("Adding SMAs...")
    df = add_sma(df, fast_period=10, slow_period=30)
    print("Plotting SMAs...")
    plot_sma(df, ticker, output_folder)
    
    print("Assigning investment signals...")
    df['is_invested'] = assign_is_invested(df)
    df = calculate_algo_log_return(df)
    
    print("Performing grid search for best SMA parameters...")
    best_fast, best_slow, _ = grid_search(df)
    print("Calculating statistics...")
    stats = calculate_statistics(df)
    
    print(f"Best Fast Period: {best_fast}, Best Slow Period: {best_slow}")
    print(f"Train Sharpe Ratio (Algo): {stats[0][0]}, Test Sharpe Ratio (Algo): {stats[0][1]}")
    print(f"Train Sharpe Ratio (Buy & Hold): {stats[1][0]}, Test Sharpe Ratio (Buy & Hold): {stats[1][1]}")
    
    print("Plotting wealth over time...")
    plot_wealth(df, output_folder)
    
    print("Saving dataframe to CSV...")
    df.to_csv(os.path.join(output_folder, 'spy_data.csv'), index=False)
    print("All tasks completed.")
