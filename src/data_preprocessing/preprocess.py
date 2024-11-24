import pandas as pd
import pandas as pd


class Preproccesor:

    def __init__(self, df: pd.DataFrame, target_column: str = "TARGET"):
        self._target = df[target_column]
        self._features = df.drop(target_column, axis=1)

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target

    def remove_zero_variance_columns(self):
        selected_col = []
        for col_name in self._features.columns:
            col = self._features[col_name]
            self.add_non_zero_variance_col(selected_col, col)
        print(len(selected_col))
        return self._features[selected_col]

    def add_non_zero_variance_col(self, selected_col: list, col: pd.Series):

        if col.var() != 0:
            return selected_col.append(col)
        else:
            return selected_col


Preproccesor(df=pd.read_csv("data/train.csv")).remove_zero_variance_columns()
