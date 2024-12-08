from src.data_preprocessing.data_object import Data
from .correlated_features import select_correlated_representative_feature


def select_representative_feature():
    data_train = Data(target_column="TARGET")
    data_test = Data(name="test.csv")
    print(len(data_train.features.columns))
    select_correlated_representative_feature(data_train, data_test, threshold=0.9)
    print(len(data_train.features.columns))


select_representative_feature()
