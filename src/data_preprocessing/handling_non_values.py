import pandas as pd

from data_object import Data


def handle_nan(data_train: Data, data_test: Data, nan_threshold=0.5):
    # TODO add other imputing methods instead of median
    numeric_columns = data_train.numerical_columns
    numeric_columns, _ = remove_high_nan_features(
        data_train, data_test, numeric_columns, nan_threshold
    )
    handle_numerical_nan(data_train, data_test, numeric_columns, nan_threshold)

    cat_bool_columns = data_train.boolean_columns
    cat_bool_columns.extend(data_train.categorical_columns)
    cat_bool_columns, _ = remove_high_nan_features(
        data_train, data_test, cat_bool_columns, nan_threshold
    )
    handle_categorical_nan(data_train, data_test, cat_bool_columns, nan_threshold)


def handle_numerical_nan(
    data_train: Data, data_test: Data, numeric_columns, nan_threshold
):
    for column in numeric_columns:
        if data_train.features[column].isnull().any():
            median_value = data_train.features[column].median()
            nan_indices = data_train.features[column].isna().index.tolist()
            data_train.set_new_values(column, nan_indices, median_value)


def handle_categorical_nan(
    data_train: Data, data_test: Data, cat_bool_columns, nan_threshold
):

    for column in cat_bool_columns:
        if data_train.features[column].isnull().any():
            new_category = "New_category"
            nan_indices = data_train.features[
                data_train.features[column].isna()
            ].index.tolist()
            data_train.set_new_values(column, nan_indices, new_category)


def remove_high_nan_features(
    data_train: Data, data_test: Data, columns_name, nan_threshold
):
    nan_proportions = data_train.features[columns_name].isna().mean()

    columns_to_keep, removed_columns_info = get_removed_kept_columns(
        data_train, data_test, nan_proportions, nan_threshold, columns_name
    )

    return columns_to_keep, removed_columns_info


def get_removed_kept_columns(
    data_train: Data, data_test: Data, nan_proportions, nan_threshold, columns_name
):
    if nan_threshold is not None:
        columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
        columns_to_keep = nan_proportions[nan_proportions <= nan_threshold].index
    else:
        columns_to_remove = []
        columns_to_keep = columns_name
    removed_columns_info = pd.DataFrame(
        {
            "column": columns_to_remove,
            "nan_proportion": nan_proportions[columns_to_remove],
        }
    )
    data_train.remove_columns(columns_to_remove)
    data_test.remove_columns(columns_to_remove)
    return columns_to_keep, removed_columns_info
