import os
from q_learning_trader.q_learning_trader import QLearningTrader
from regression_based_strategy.main import run_regression_strategy
from trend_following_sma.trend_following_sma import run_trend_following_sma
from trend_following_reinforcement_learning.main import run_reinforcement_learning

def run_q_learning_trader():
    trader = QLearningTrader('AAPL', '2010-01-01', '2020-01-01')
    trader.train()
    final_value = trader.simulate_trading()
    print(f"Q-Learning Trader Final portfolio value: ${final_value:.2f}")
    trader.plot_returns()
    trader.export_results(final_value)

def run_all_strategies():
    print("Running Q-Learning Trader...")
    run_q_learning_trader()

    print("Running Regression-Based Strategy...")
    run_regression_strategy()

    print("Running Trend Following SMA Strategy...")
    run_trend_following_sma()

    print("Running Trend Following Reinforcement Learning Strategy...")
    run_reinforcement_learning()

if __name__ == "__main__":
    run_all_strategies()
