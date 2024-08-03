def create_shifted_target(df_returns, target_col='SPY'):
    df_returns[target_col] = df_returns[target_col].shift(-1)
    return df_returns

def split_data(df_returns, Ntest, target_col='SPY', x_cols=None):
    train = df_returns.iloc[1:-Ntest]
    test = df_returns.iloc[-Ntest:-1]
    
    if x_cols is None:
        x_cols = df_returns.columns.drop(target_col)
    
    Xtrain = train[x_cols]
    Ytrain = train[target_col]
    Xtest = test[x_cols]
    Ytest = test[target_col]
    
    return Xtrain, Ytrain, Xtest, Ytest
