import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from tqdm import tqdm

class QLearningTrader:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.load_data()
        self.q_table = None

    def load_data(self):
        # Load stock data from Yahoo Finance
        data = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        data['Returns'] = data['Adj Close'].pct_change().fillna(0)
        return data

    def train(self, episodes=20, alpha=0.1, gamma=0.9, epsilon=0.2):
        # Initialize Q-table with float64 data type
        states = self.data['Returns'].apply(lambda x: int(x * 100)).unique()
        self.q_table = pd.DataFrame(0.0, index=states, columns=['Hold', 'Buy', 'Sell'])

        for _ in tqdm(range(episodes), desc="Training Episodes"):
            state = self.get_state(0)
            for t in range(len(self.data) - 1):
                if np.random.rand() < epsilon:
                    action = np.random.choice(['Hold', 'Buy', 'Sell'])
                else:
                    action = self.q_table.loc[state].idxmax()

                reward = self.get_reward(t, action)
                next_state = self.get_state(t + 1)

                self.q_table.loc[state, action] = (1 - alpha) * self.q_table.loc[state, action] + alpha * (
                    reward + gamma * self.q_table.loc[next_state].max())

                state = next_state

    def get_state(self, t):
        return int(self.data['Returns'].iloc[t] * 100)

    def get_reward(self, t, action):
        if action == 'Buy':
            return self.data['Adj Close'].iloc[t + 1] - self.data['Adj Close'].iloc[t]
        elif action == 'Sell':
            return self.data['Adj Close'].iloc[t] - self.data['Adj Close'].iloc[t + 1]
        else:
            return 0

    def simulate_trading(self):
        state = self.get_state(0)
        portfolio = 10000
        shares = 0

        for t in range(len(self.data) - 1):
            action = self.q_table.loc[state].idxmax()

            if action == 'Buy' and portfolio > 0:
                shares += portfolio // self.data['Adj Close'].iloc[t]
                portfolio %= self.data['Adj Close'].iloc[t]
            elif action == 'Sell' and shares > 0:
                portfolio += shares * self.data['Adj Close'].iloc[t]
                shares = 0

            state = self.get_state(t + 1)

        final_value = portfolio + shares * self.data['Adj Close'].iloc[-1]
        return final_value

    def plot_returns(self):
        self.data['Strategy'] = self.data['Returns'].cumsum()
        self.data['Buy and Hold'] = np.log1p(self.data['Returns']).cumsum()
        self.data[['Strategy', 'Buy and Hold']].plot()
        plt.title('Trading Strategy vs. Buy and Hold')
        plt.savefig('returns_plot.png')  # Save the plot as an image
        plt.show()

    def export_results(self, final_value):
        with open('results.txt', 'w') as f:
            f.write(f"Final portfolio value: ${final_value:.2f}\n")
            f.write("Q-Table:\n")
            f.write(self.q_table.to_string())

def run_q_learning_trader():
    trader = QLearningTrader('AAPL', '2010-01-01', '2024-08-03')
    trader.train()
    final_value = trader.simulate_trading()
    print(f"Final portfolio value: ${final_value:.2f}")
    trader.plot_returns()
    trader.export_results(final_value)
