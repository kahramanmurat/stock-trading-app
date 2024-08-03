# main.py
import pandas as pd
import numpy as np
from data_module import fetch_data, preprocess_data
from feature_engineering import create_features
from training_evaluation import train_and_evaluate
from utils import plot_data, plot_returns

def main():
    ticker_symbol = 'SPY'
    df = fetch_data(ticker_symbol)
    plot_data(df)
    df = preprocess_data(df)
    df, feats = create_features(df)

    Ntest = 1000
    train_data = df.iloc[:-Ntest].copy()
    test_data = df.iloc[-Ntest:].copy()

    train_reward, test_reward, train_buy_and_hold, test_buy_and_hold, train_data, test_data = train_and_evaluate(train_data, test_data, feats)
    
    print(f'Train Reward: {train_reward}, Train Buy and Hold: {train_buy_and_hold}')
    print(f'Test Reward: {test_reward}, Test Buy and Hold: {test_buy_and_hold}')

    plot_returns(train_data, 'Train Data Cumulative Log Return', 'train_cumulative_return.png')
    plot_returns(test_data, 'Test Data Cumulative Log Return', 'test_cumulative_return.png')

if __name__ == "__main__":
    main()
