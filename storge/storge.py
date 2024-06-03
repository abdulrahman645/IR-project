import pickle
import io
import os
import pandas as pd
from pathlib import Path






cache = {}

def load_binary_file(dataset: str, var_type: str):
    path = f'./storge/binaryfiles/{dataset}_{var_type}.pkl'
    path = Path(path)
    if path in cache:
        return cache[path]
    else:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            cache[path] = obj
        return obj



def save_binary_file(dataset: str, var_type: str, var):
    path = f'./storge/binaryfiles/{dataset}_{var_type}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(var, f)



df_cache = {}

def load_df(dataset, var_type):
    file_path = f'./storge/datasets/{dataset}/{dataset}_{var_type}.csv'
    file_path = Path(file_path)
    if file_path in df_cache:
        return df_cache[file_path]
    else:
        df = pd.read_csv(file_path)
        df_cache[file_path] = df
        return df