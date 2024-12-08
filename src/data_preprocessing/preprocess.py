import pandas as pd


from data_object import Data
from handling_no_variance import handling_no_variance


# TODO: Change the Preproccesor class to a modul and separate variance based feature removing and nan value handling


def preprocess():
    data_train = Data(target_column="TARGET")
    data_test = Data(name="test.csv")
    handling_no_variance(data_train, data_test)


preprocess()
