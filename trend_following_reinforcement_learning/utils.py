# utils.py
import matplotlib.pyplot as plt

def plot_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Adj Close'], label='Adjusted Close Price')
    plt.title('SPY Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_returns(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['LogReturn'].cumsum(), label='Cumulative Log Return')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
