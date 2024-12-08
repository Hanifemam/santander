# def handel_nan_with(self, nan_threshold=0.5):
#     # TODO add other imputing methods instead of median
#     numeric_columns = self.data_train.numerical_columns
#     self.handle_numerical_nan(numeric_columns, nan_threshold)

#     cat_bool_columns = self.data_train.boolean_columns
#     cat_bool_columns.extend(self.data_train.categorical_columns)
#     self.handle_bool_categorical_nan(cat_bool_columns, nan_threshold)


# def handle_categorical_nan(self, numeric_columns, nan_threshold):
#     columns_to_keep, _ = self.remove_high_nan_features(numeric_columns, nan_threshold)
#     for column in columns_to_keep:
#         if self._features_train[column].isnull().any():
#             median_value = "New category"
#             nan_indices = self._features_train[
#                 self._features_train[column].isna()
#             ].index.tolist()
#             self.data_test.set_new_values(column, nan_indices, median_value)


# def handle_numerical_nan(self, numeric_columns, nan_threshold):
#     columns_to_keep, _ = self.remove_high_nan_features(numeric_columns, nan_threshold)
#     for column in columns_to_keep:
#         if self._features_train[column].isnull().any():
#             median_value = self._features_train[column].median()
#             nan_indices = self._features_train[
#                 self._features_train[column].isna()
#             ].index.tolist()
#             self.data_test.set_new_values(column, nan_indices, median_value)


# def remove_high_nan_features(self, columns_name, nan_threshold):
#     nan_proportions = self._features_train[columns_name].isna().mean()

#     if nan_threshold is not None:
#         columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
#         columns_to_keep = nan_proportions[nan_proportions <= nan_threshold].index
#     else:
#         columns_to_remove = []
#         columns_to_keep = columns_name
#     self._features_train = self.data_train.remove_columns(columns_to_remove)
#     removed_columns_info = pd.DataFrame(
#         {
#             "column": columns_to_remove,
#             "nan_proportion": nan_proportions[columns_to_remove],
#         }
#     )
#     return columns_to_keep, removed_columns_info
