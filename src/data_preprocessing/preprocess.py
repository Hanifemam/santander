import pandas as pd


from data_object import Data


class Preproccesor:

    def __init__(self):
        self.data_train = Data(target_column="TARGET")
        self.data_test = Data(name="test.csv")
        self._target = self.data_train.target
        self._features_train = self.data_train.features
        self._features_test = self.data_test.features

    def remove_zero_variance_columns(self):
        selected_col = []
        for col_name in self._features_train.columns:
            col = self._features_train[col_name]
            self.add_zero_variance_col(selected_col, col)
        self.data_train.remove_columns(selected_col)
        self.data_test.remove_columns(selected_col)

    def add_zero_variance_col(self, selected_col: list, col: pd.Series):
        if col.var() == 0:
            return selected_col.append(col.name)
        else:
            return selected_col

    def remove_feature_class_without_var(self):
        # TODO refactor
        class_based_zero_variance = dict()
        df_positive, df_negative = self.data_train.get_positive_negative_classes()
        for column in df_positive.columns:
            if len(df_positive[column].unique()) != 1:

                if len(df_negative[column].unique()) == 1:
                    class_based_zero_variance[column] = (
                        0,
                        float(df_negative[column].unique()[0]),
                    )
            else:
                if len(df_negative[column].unique()) != 1:
                    class_based_zero_variance[column] = (
                        1,
                        float(df_positive[column].unique()[0]),
                    )
        return class_based_zero_variance


data = Preproccesor()
data.remove_zero_variance_columns()
data.remove_feature_class_without_var()
print(data._features_train.head())
