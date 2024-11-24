import pandas as pd


class Data:

    dir_ = "data/"

    def __init__(
        self, df: pd.DataFrame = None, name="train.csv", target_column: str = None
    ):
        self.file_name = name
        if df is None:
            self._df = self.load_data(Data.dir_ + name)
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

    def save_data(self, name="temp_data.csv"):
        df = self.concat_data_frames()
        if name.endswith("csv"):
            df.to_csv(Data.dir_ + name)
        else:
            df.to_pickle(Data.dir_ + name)

    def concat_data_frames(self):
        if self._target is None:
            return self.features
        else:
            return pd.concat([self._features, self._features], axis=1)

    def remove_columns(self, removing_features: list):
        return self._features[removing_features]
