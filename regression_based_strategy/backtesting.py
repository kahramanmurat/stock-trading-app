import numpy as np

def backtest(df_returns, Ptrain, Ptest, train_idx, test_idx, target_col='SPY'):
    df_returns['Position'] = 0
    df_returns.loc[train_idx, 'Position'] = (Ptrain > 0).astype(int)
    df_returns.loc[test_idx, 'Position'] = (Ptest > 0).astype(int)
    df_returns['AlgoReturn'] = df_returns['Position'] * df_returns[target_col]
    return df_returns

def calculate_returns(df_returns, Ntest):
    train_return = df_returns.iloc[1:-Ntest]['AlgoReturn'].sum()
    test_return = df_returns.iloc[-Ntest:-1]['AlgoReturn'].sum()
    return train_return, test_return
