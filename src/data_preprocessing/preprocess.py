import pandas as pd


from data_object import Data


class Preproccesor:

    def __init__(self):
        self.data = Data()
        self._target = data.target
        self._features = data.features

    def remove_zero_variance_columns(self):
        selected_col = []
        for col_name in self._features.columns:
            col = self._features[col_name]
            self.add_non_zero_variance_col(selected_col, col)

        self.data.remove_columns(selected_col)

    def add_non_zero_variance_col(self, selected_col: list, col: pd.Series):

        if col.var() != 0:
            return selected_col.append(col)
        else:
            return selected_col

    def remove_feature_class_without_var(self):
        pass


Preproccesor(df=pd.read_csv("data/train.csv")).remove_zero_variance_columns()
