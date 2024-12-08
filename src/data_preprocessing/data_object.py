import pandas as pd


class Data:
    # TODO set an attribute setter function
    dir_ = "data/"

    def __init__(
        self,
        df: pd.DataFrame = None,
        name="train.csv",
        target_column: str = None,
        id_columns: list[str] = ["ID"],
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

        for id in id_columns:
            if id in self.features.columns:
                self._features = self._features.drop(id, axis=1)

        self._numerical_columns = None
        self._categorical_columns = None
        self._boolean_columns = None

        self.set_data_type(self.get_data_type())

    def __len__(self):
        return len(self._features.columns)

    @property
    def numerical_columns(self):
        return self._numerical_columns

    @property
    def categorical_columns(self):
        return self._categorical_columns

    @property
    def boolean_columns(self):
        return self._boolean_columns

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
        self._features = self._features.drop(removing_features, axis=1)
        self.set_data_type(self.get_data_type())
        return self._features

    def get_positive_negative_classes(self):
        df_positive = self._features[self._target == 1]
        df_negative = self._features[self._target == 0]
        return df_positive, df_negative

    def get_data_type(self):
        return {
            "numbererical": self._features.select_dtypes(
                include=["number"]
            ).columns.tolist(),
            "categorical": self._features.select_dtypes(
                include=["object"]
            ).columns.tolist(),
            "boolean": self._features.select_dtypes(include=["bool"]).columns.tolist(),
        }

    def set_data_type(self, data_type_list):
        self._numerical_columns = data_type_list["numbererical"]
        self._categorical_columns = data_type_list["categorical"]
        self._boolean_columns = data_type_list["boolean"]

    def set_new_values(self, feature, index_list, new_value):
        self._features.loc[index_list, feature] = new_value

    # TODO change features to df_features
