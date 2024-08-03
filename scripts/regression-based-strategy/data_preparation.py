import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def prepare_returns(df):
    log_diff_list = [np.log(df[name]).diff() for name in df.columns]
    df_returns = pd.concat(log_diff_list, axis=1)
    df_returns.columns = df.columns
    return df_returns
