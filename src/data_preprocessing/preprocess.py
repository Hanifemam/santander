import pandas as pd


from data_object import Data
from handling_no_variance import handle_no_variance
from handling_non_values import handle_nan
from data_transforming import data_transform


def preprocess():
    data_train = Data(target_column="TARGET")
    data_test = Data(name="test.csv")
    handle_no_variance(data_train, data_test)
    handle_nan(data_train, data_test)
    data_transform(data_train, data_test)


preprocess()
