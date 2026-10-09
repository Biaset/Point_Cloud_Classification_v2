"""
Feature engineering for point-cloud classification.

The contextual features reproduce the KNN enhancement used in main_2:
for each spatial/photometric feature, compute neighborhood mean, standard
deviation, 90th percentile and point-minus-neighborhood-mean.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from config import N_NEIGHBORS, SPATIAL_FEATURES, TILE_SIZE


def validate_feature_columns(df):
    missing = [column for column in SPATIAL_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(
            "The following required feature columns are missing from the input: "
            + ", ".join(missing)
        )


def build_features(df, n_neighbors=N_NEIGHBORS):
    """
    Build the final feature matrix:
    XYZ + original spatial/photometric features + KNN contextual features.
    """
    validate_feature_columns(df)

    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)
    spatial_features = df[SPATIAL_FEATURES].to_numpy(dtype=np.float32)

    k = min(n_neighbors, len(df))
    if k < 1:
        raise ValueError("The point cloud contains no points.")

    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree")
    nn.fit(xyz)
    _, indices = nn.kneighbors(xyz)

    contextual_features = []

    for index, column in enumerate(SPATIAL_FEATURES):
        values = spatial_features[:, index]
        neighbors = values[indices]

        neighborhood_mean = neighbors.mean(axis=1)
        neighborhood_std = neighbors.std(axis=1)
        neighborhood_p90 = np.percentile(neighbors, 90, axis=1)
        local_difference = values - neighborhood_mean

        contextual_features.extend(
            [neighborhood_mean, neighborhood_std, neighborhood_p90, local_difference]
        )

    context = np.column_stack(contextual_features).astype(np.float32)

    return np.column_stack([xyz, spatial_features, context]).astype(np.float32)


def build_tile_ids(xyz, tile_size=TILE_SIZE):
    """Create deterministic spatial groups for grouped cross-validation."""
    xyz = np.asarray(xyz)
    origin = xyz.min(axis=0)
    ijk = np.floor((xyz - origin) / tile_size).astype(np.int32)

    return (
        ijk[:, 0]
        + 73856093 * ijk[:, 1]
        + 19349663 * ijk[:, 2]
    )


def prepare_train_test_features(df, train_clusters, test_clusters):
    """Create feature matrices and labels using whole clusters for the split."""
    train_mask = df["Cluster"].isin(train_clusters).to_numpy()
    test_mask = df["Cluster"].isin(test_clusters).to_numpy()

    X_all = build_features(df)
    y_all = df["Classification"].to_numpy(dtype=int)

    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    y_train = y_all[train_mask]
    y_test = y_all[test_mask]

    groups = build_tile_ids(df.loc[train_mask, ["X", "Y", "Z"]].to_numpy())

    return X_train, X_test, y_train, y_test, groups
