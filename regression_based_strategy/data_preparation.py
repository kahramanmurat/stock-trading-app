import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    return df

def prepare_returns(df):
    df_returns = df.pct_change().apply(np.log1p)
    return df_returns
