import pandas as pd


def load_data(name="train"):
    return pd.read_csv(f"data/{name}.csv")