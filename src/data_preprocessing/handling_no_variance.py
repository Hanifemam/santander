import pandas as pd

from data_object import Data


def handling_no_variance(data_train: Data, data_test: Data):
    remove_zero_variance_columns(data_train, data_test)
    remove_feature_class_without_var(data_train)


def remove_zero_variance_columns(data_train: Data, data_test: Data):
    selected_col = []
    for col_name in data_train.features.columns:
        col = data_train.features[col_name]
        add_zero_variance_col(selected_col, col)
    data_train.remove_columns(selected_col)
    data_test.remove_columns(selected_col)


def add_zero_variance_col(selected_col: list, col: pd.Series):
    if col.var() == 0:
        return selected_col.append(col.name)
    else:
        return selected_col


def remove_feature_class_without_var(data: Data):
    # TODO refactor
    class_based_zero_variance = dict()
    df_positive, df_negative = data.get_positive_negative_classes()
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
