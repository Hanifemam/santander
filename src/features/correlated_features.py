import pandas as pd
import numpy as np
import networkx as nx

from src.data_preprocessing.data_object import Data


def select_correlated_representative_feature(
    data_train: Data, data_test: Data, threshold=0.9
):
    correlated_feature_groups = identify_correlated_groups(data_train, threshold)
    none_representative_features = get_representative_features_list(
        correlated_feature_groups, data_train
    )
    data_train.remove_columns(none_representative_features)
    data_test.remove_columns(none_representative_features)


def identify_correlated_groups(data: Data, threshold=0.9):
    corr_matrix = data.features.corr(method="spearman").abs()
    corr_matrix = corr_matrix.copy()
    np.fill_diagonal(corr_matrix.values, 0)
    adj_matrix = (corr_matrix.abs() >= threshold).astype(int)
    G = nx.from_pandas_adjacency(adj_matrix)
    none_representative_features = list(nx.connected_components(G))

    return none_representative_features


def get_representative_features_list(groups, data: Data):
    representative_features = []
    for group in groups:
        group = list(group)
        variances = data.features[group].var()
        representative_feature = variances.idxmax()
        representative_features.append(representative_feature)
    all_grouped_features = set().union(*groups)
    none_representative_features = [
        feature
        for feature in all_grouped_features
        if feature not in representative_features
    ]

    return none_representative_features
