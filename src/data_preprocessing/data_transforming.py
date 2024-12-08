import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

from data_object import Data


def data_transform(data_train: Data, data_test: Data):
    power_transform(data_train, data_test)


def power_transform(data_train: Data, data_test: Data):
    pt_fit = PowerTransformer().fit(data_train.features)
    data_train.transform_data(pt_fit)
    data_test.transform_data(pt_fit)
