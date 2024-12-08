import pandas as pd

from data_object import Data


def handle_nan(data_train: Data, data_test: Data, nan_threshold=0.5):
    removed_column_info = remove_high_nan_features(data_train, data_test, nan_threshold)
    handle_numerical_nan(data_train, data_test)
    handle_categorical_nan(data_train, data_test)
    return removed_column_info


def handle_numerical_nan(data_train: Data, data_test: Data):
    numeric_columns = data_train.numerical_columns
    for column in numeric_columns:
        median_value = data_train.features[column].median()
        set_values_for_numeric_nan(data_train, column, median_value)
        set_values_for_numeric_nan(data_test, column, median_value)


def set_values_for_numeric_nan(data: Data, column, median_value):
    nan_indices = data.features[column].isna().index.tolist()
    data.set_new_values(column, nan_indices, median_value)


def handle_categorical_nan(data_train: Data, data_test: Data):
    cat_bool_columns = data_train.boolean_columns
    cat_bool_columns.extend(data_train.categorical_columns)
    for column in cat_bool_columns:
        if data_train.features[column].isnull().any():
            new_category = "New_category"
            nan_indices = data_train.features[
                data_train.features[column].isna()
            ].index.tolist()
            data_train.set_new_values(column, nan_indices, new_category)


def remove_high_nan_features(data_train: Data, data_test: Data, nan_threshold):
    columns_name = data_train.features.columns
    nan_proportions = data_train.features[columns_name].isna().mean()

    removed_columns_info_retured = removed_columns_info(
        data_train, data_test, nan_proportions, nan_threshold
    )

    return removed_columns_info_retured


def removed_columns_info(
    data_train: Data, data_test: Data, nan_proportions, nan_threshold
):
    if nan_threshold is not None:
        columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
    else:
        columns_to_remove = []
    removed_columns_info = pd.DataFrame(
        {
            "column": columns_to_remove,
            "nan_proportion": nan_proportions[columns_to_remove],
        }
    )
    data_train.remove_columns(columns_to_remove)
    data_test.remove_columns(columns_to_remove)
    return removed_columns_info
