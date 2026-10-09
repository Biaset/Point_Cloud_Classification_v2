"""
Model registry and hyperparameter spaces.

To add a new model:
1. Add its estimator to MODEL_REGISTRY.
2. Add its search space to SEARCH_SPACES.
3. If it needs special fit parameters, handle them in get_fit_params().
"""

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42

MODEL_ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "random_forest": "random_forest",
    "et": "extra_trees",
    "extratrees": "extra_trees",
    "extra_trees": "extra_trees",
    "xgb": "xgboost",
    "xgboost": "xgboost",
}

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
    ),
    "extra_trees": ExtraTreesClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=1,
    ),
}


def build_xgboost(n_classes):
    """Create the XGBoost estimator used by the original main_2 workflow."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
    )


# Hyperparameter spaces retained from the original project.
# Keys are prefixed with "model__" because the estimators are inside a Pipeline.
SEARCH_SPACES = {
    "random_forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [20, 30, None],
        "model__min_samples_split": [2, 4],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 4],
    },
    "extra_trees": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [15, 20, 25],
        "model__min_samples_leaf": [200],
        "model__max_features": [0.3],
        "model__min_samples_split": [500],
    },
    "xgboost": {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [3, 5, 7],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__gamma": [0, 0.1, 0.3],
        "model__learning_rate": [0.01, 0.05, 0.1],
    },
}


# Reference configurations recorded in the original project.
# They are kept here for reproducibility and can be used directly if desired.
REFERENCE_PARAMS = {
    "xgboost": {
        "n_estimators": 800,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.1,
        "learning_rate": 0.1,
    },
    "extra_trees": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 10,
        "max_features": 0.5,
    },
}


def normalize_model_name(model_name):
    name = model_name.lower()
    if name not in MODEL_ALIASES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {sorted(set(MODEL_ALIASES.values()))}"
        )
    return MODEL_ALIASES[name]


def get_model(model_name, n_classes):
    """Return an unfitted estimator for the requested model."""
    name = normalize_model_name(model_name)

    if name == "xgboost":
        return build_xgboost(n_classes)

    return MODEL_REGISTRY[name]


def build_pipeline(model_name, n_classes):
    """
    Build the common training pipeline.

    StandardScaler is retained to keep the workflow compatible with main_2.
    Tree models do not require scaling, but leaving the step in the pipeline
    makes model interchangeability straightforward.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", get_model(model_name, n_classes)),
    ])


def get_search_space(model_name):
    name = normalize_model_name(model_name)
    return SEARCH_SPACES[name]


def get_fit_params(model_name, sample_weights):
    """Return model-specific fit parameters."""
    if normalize_model_name(model_name) == "xgboost":
        return {"model__sample_weight": sample_weights}
    return {}
