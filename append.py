import pandas as pd
from glob import glob
import os

# Get all CSV files in the 'data' directory
files = glob('data/*.csv')

data_frames = []
for f in files:
    print(f)
    df = pd.read_csv(f)
    
    # Extract symbol from the filename
    symbol = os.path.basename(f).split('.')[0]
    df['Name'] = symbol
    
    data_frames.append(df)

# Concatenate all DataFrames
full_df = pd.concat(data_frames, ignore_index=True)

# Save the final DataFrame to a CSV file
full_df.to_csv('sp500full.csv', index=False)
