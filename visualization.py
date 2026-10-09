"""
Point-cloud classification visualization utilities.
"""

import numpy as np
import pandas as pd


def add_classification_colors(df, class_colors):
    """
    Add Cls_Rf/Cls_Gf/Cls_Bf columns based on predicted Classification.
    Original RGB values are retained for classes mapped to "original_color".
    """
    result = df.copy()
    result[["Cls_Rf", "Cls_Gf", "Cls_Bf"]] = 0.0

    for label, color in class_colors.items():
        mask = result["Classification"].astype(float) == float(label)

        if color == "original_color":
            result.loc[mask, ["Cls_Rf", "Cls_Gf", "Cls_Bf"]] = (
                result.loc[mask, ["Rf", "Gf", "Bf"]].to_numpy()
            )
        else:
            result.loc[mask, ["Cls_Rf", "Cls_Gf", "Cls_Bf"]] = color

    return result


def point_cloud_arrays(df):
    """Return XYZ coordinates and RGB colors for Open3D."""
    xyz = df[["X", "Y", "Z"]].to_numpy()
    rgb = df[["Cls_Rf", "Cls_Gf", "Cls_Bf"]].to_numpy()
    return xyz, rgb


def build_prediction_dataframe(X, predictions):
    """
    Convert a feature matrix produced by build_features() into the compact
    point-cloud format used for visualization/export.
    """
    result = pd.DataFrame(
        {
            "X": X[:, 0],
            "Y": X[:, 1],
            "Z": X[:, 2],
            "Rf": X[:, 3],
            "Gf": X[:, 4],
            "Bf": X[:, 5],
            "Intensity": X[:, 6],
            "Nx": X[:, 7],
            "Ny": X[:, 8],
            "Nz": X[:, 9],
            "Classification": predictions,
        }
    )
    return result
