import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .data_preparation import load_data, prepare_returns
from .feature_engineering import create_shifted_target, split_data
from .model_training import (
    train_linear_regression,
    train_logistic_regression,
    train_random_forest,
    evaluate_regression_model,
    evaluate_classification_model,
    predict
)
from .backtesting import backtest, calculate_returns

def export_plot_returns(df_returns, Ntest, strategy_name, output_file):
    train_returns = df_returns.iloc[1:-Ntest]['AlgoReturn'].cumsum()
    test_returns = df_returns.iloc[-Ntest:-1]['AlgoReturn'].cumsum()
    buy_hold_train_returns = df_returns.iloc[1:-Ntest]['SPY'].cumsum()
    buy_hold_test_returns = df_returns.iloc[-Ntest:-1]['SPY'].cumsum()

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(train_returns, label=f'{strategy_name} Strategy Return (Train)')
    plt.plot(buy_hold_train_returns, label='Buy and Hold Return (Train)')
    plt.legend()
    plt.title('Training Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Log Return')

    plt.subplot(1, 2, 2)
    plt.plot(test_returns, label=f'{strategy_name} Strategy Return (Test)')
    plt.plot(buy_hold_test_returns, label='Buy and Hold Return (Test)')
    plt.legend()
    plt.title('Testing Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Log Return')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def run_regression_strategy():
    # Parameters
    file_path = 'sp500_closefull.csv'
    Ntest = 1000
    target_col = 'SPY'
    x_cols = ['AAPL', 'MSFT', 'AMZN', 'JNJ', 'V', 'PG', 'JPM']

    # Data Preparation
    df = load_data(file_path)
    df_returns = prepare_returns(df)
    df_returns = create_shifted_target(df_returns, target_col)

    # Split Data
    Xtrain, Ytrain, Xtest, Ytest = split_data(df_returns, Ntest, target_col, x_cols)

    # Train Models
    print("Training Linear Regression Model...")
    linear_model = train_linear_regression(Xtrain, Ytrain)

    print("Training Logistic Regression Model...")
    logistic_model = train_logistic_regression(Xtrain, Ytrain)

    print("Training Random Forest Model...")
    rf_model = train_random_forest(Xtrain, Ytrain)

    # Evaluate Models
    print("Evaluating Models...")
    print("Linear Regression:", evaluate_regression_model(linear_model, Xtrain, Ytrain, Xtest, Ytest))
    print("Logistic Regression:", evaluate_classification_model(logistic_model, Xtrain, Ytrain > 0, Xtest, Ytest > 0))
    print("Random Forest:", evaluate_classification_model(rf_model, Xtrain, Ytrain > 0, Xtest, Ytest > 0))

    # Predictions
    print("Generating Predictions...")
    Ptrain_lr, Ptest_lr = predict(linear_model, Xtrain, Xtest)
    Ptrain_log, Ptest_log = predict(logistic_model, Xtrain, Xtest)
    Ptrain_rf, Ptest_rf = predict(rf_model, Xtrain, Xtest)

    # Backtesting
    train_idx = df.index <= df_returns.index[-Ntest-1]
    test_idx = df.index > df_returns.index[-Ntest-1]
    train_idx[0] = False
    test_idx[-1] = False

    # Backtest Linear Regression
    print("Backtesting Linear Regression Model...")
    df_returns_lr = backtest(df_returns.copy(), Ptrain_lr, Ptest_lr, train_idx, test_idx, target_col)
    
    # Backtest Logistic Regression
    print("Backtesting Logistic Regression Model...")
    df_returns_log = backtest(df_returns.copy(), Ptrain_log, Ptest_log, train_idx, test_idx, target_col)
    
    # Backtest Random Forest
    print("Backtesting Random Forest Model...")
    df_returns_rf = backtest(df_returns.copy(), Ptrain_rf, Ptest_rf, train_idx, test_idx, target_col)

    # Export plots
    print("Exporting Plot Returns...")
    export_plot_returns(df_returns_lr, Ntest, 'Linear Regression', 'plots/linear_regression_returns.png')
    export_plot_returns(df_returns_log, Ntest, 'Logistic Regression', 'plots/logistic_regression_returns.png')
    export_plot_returns(df_returns_rf, Ntest, 'Random Forest', 'plots/random_forest_returns.png')

if __name__ == "__main__":
    run_regression_strategy()
