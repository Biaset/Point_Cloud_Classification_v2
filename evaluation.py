"""
Training, evaluation and prediction utilities.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from config import N_CV_SPLITS, N_JOBS, N_RANDOM_SEARCH_ITERATIONS, RESULTS_DIR
from models import build_pipeline, get_fit_params, get_search_space


def train_model(
    model_name,
    X_train,
    y_train,
    groups,
    n_classes,
    random_search=True,
    n_iter=N_RANDOM_SEARCH_ITERATIONS,
):
    """Tune and fit the selected model using grouped cross-validation."""
    pipeline = build_pipeline(model_name, n_classes)
    search_space = get_search_space(model_name)
    cv = GroupKFold(n_splits=N_CV_SPLITS)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    fit_params = get_fit_params(model_name, sample_weights)

    if random_search:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            refit="f1",
            verbose=2,
            n_jobs=N_JOBS,
            random_state=42,
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=search_space,
            scoring=scoring,
            cv=cv,
            refit="f1",
            verbose=2,
            n_jobs=N_JOBS,
        )

    search.fit(X_train, y_train, groups=groups, **fit_params)
    return search


def evaluate_model(model, X_test, y_test, class_names=None):
    """Compute and print standard multiclass classification metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print("\nClassification report:")
    if class_names:
        report_labels = sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))
        report_names = [class_names.get(int(label), str(label)) for label in report_labels]
        print(
            classification_report(
                y_test,
                y_pred,
                labels=report_labels,
                target_names=report_names,
                zero_division=0,
            )
        )
    else:
        print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }

    return y_pred, metrics


def save_model(model, model_name, output_dir=RESULTS_DIR):
    """Save the fitted pipeline as a joblib file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{model_name}_pipeline.joblib"
    joblib.dump(model, path)
    return path


def save_metrics(model_name, metrics, output_dir=RESULTS_DIR):
    """Append the evaluation metrics to a CSV file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "model_metrics.csv"
    row = pd.DataFrame([{"Model": model_name, **metrics}])

    if path.exists():
        previous = pd.read_csv(path)
        row = pd.concat([previous, row], ignore_index=True)

    row.to_csv(path, index=False)
    return path


def print_confusion_matrix(y_test, y_pred):
    """Print a numeric confusion matrix."""
    labels = np.unique(np.concatenate([y_test, y_pred]))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))
