import pandas as pd
import numpy as np
import networkx as nx

from ..data_preprocessing.data_object import Data


def select_representative_feature(data_train: Data, data_test: Data):
    correlated_feature_groups = identify_correlated_groups(data_train, threshold=0.9)
    none_representative_features = get_representative_features_list(
        correlated_feature_groups, data_train
    )


def identify_correlated_groups(data, threshold=0.9):
    corr_matrix = data.corr(method="spearman").abs()
    # Make a copy to prevent modifying the original matrix
    corr_matrix = corr_matrix.copy()

    # Remove self-correlations by setting diagonal to zero
    np.fill_diagonal(corr_matrix.values, 0)

    # Apply threshold to get adjacency matrix
    adj_matrix = (corr_matrix.abs() >= threshold).astype(int)

    # Create a graph from the adjacency matrix
    G = nx.from_pandas_adjacency(adj_matrix)

    # Find connected components (groups)
    none_representative_features = list(nx.connected_components(G))

    return none_representative_features


def get_representative_features_list(groups, data):
    """
    Selects one representative feature from each group.

    Parameters:
    - groups: List of sets, groups of correlated features.
    - data: DataFrame, original data containing the features.

    Returns:
    - List of selected feature names.
    """
    selected_features = []
    for group in groups:
        group = list(group)
        # Compute variance for each feature in the group
        variances = data[group].var()
        # Select the feature with the highest variance
        representative_feature = variances.idxmax()
        selected_features.append(representative_feature)

    # Get the set of all features that are not in any group
    all_grouped_features = set().union(*groups)
    remaining_features = [
        feature for feature in all_grouped_features if feature not in selected_features
    ]

    # Combine selected features with ungrouped features
    final_features = selected_features + remaining_features
    return data[final_features]
