import pandas as pd


from data_object import Data


class Preproccesor:
    # TODO: Change the Preproccesor class to a modul and separate variance based feature removing and nan value handling

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

    def handel_nan_with(self, nan_threshold=0.5):
        # TODO add other imputing methods instead of median
        numeric_columns = self.data_train.numerical_columns
        self.handle_numerical_nan(numeric_columns, nan_threshold)

        cat_bool_columns = self.data_train.boolean_columns
        cat_bool_columns.extend(self.data_train.categorical_columns)
        self.handle_bool_categorical_nan(cat_bool_columns, nan_threshold)

    def handle_bool_categorical_nan(self, numeric_columns, nan_threshold):
        columns_to_keep, _ = self.remove_high_nan_features(
            numeric_columns, nan_threshold
        )
        for column in columns_to_keep:
            if self._features_train[column].isnull().any():
                median_value = self._features_train[column].median()
                nan_indices = self._features_train[
                    self._features_train[column].isna()
                ].index.tolist()
                self.data_test.set_new_values(column, nan_indices, median_value)

    def handle_numerical_nan(self, numeric_columns, nan_threshold):
        columns_to_keep, _ = self.remove_high_nan_features(
            numeric_columns, nan_threshold
        )
        for column in columns_to_keep:
            if self._features_train[column].isnull().any():
                median_value = self._features_train[column].median()
                nan_indices = self._features_train[
                    self._features_train[column].isna()
                ].index.tolist()
                self.data_test.set_new_values(column, nan_indices, median_value)

    def remove_high_nan_features(self, columns_name, nan_threshold):
        nan_proportions = self._features_train[columns_name].isna().mean()

        if nan_threshold is not None:
            columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
            columns_to_keep = nan_proportions[nan_proportions <= nan_threshold].index
        else:
            columns_to_remove = []
            columns_to_keep = columns_name
        self._features_train = self.data_train.remove_columns(columns_to_remove)
        removed_columns_info = pd.DataFrame(
            {
                "column": columns_to_remove,
                "nan_proportion": nan_proportions[columns_to_remove],
            }
        )
        return columns_to_keep, removed_columns_info


Preproccesor().handel_nan_with_median()
# data.remove_zero_variance_columns()
# data.remove_feature_class_without_var()
