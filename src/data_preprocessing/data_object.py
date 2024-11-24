import pandas as pd


class Data:

    def __init__(
        self, df: pd.DataFrame = None, name="name.csv", target_column: str = None
    ):
        if df is None:
            dir_ = "data/" + name
            self._df = self.load_data(dir_)
        else:
            self._df = df

        if target_column is None:
            self._target = None
            self._features = self._df
        else:
            self._target = self._df[target_column]
            self._features = self._df.drop(target_column, axis=1)

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target

    def load_data(self, dir: str):
        if dir.endswith("csv"):
            return pd.read_csv(dir)
        else:
            return pd.read_pickle(dir)
