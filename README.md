# Point Cloud Classification

Machine-learning workflow for classifying pathologies in heritage-building point clouds using geometric, photometric and KNN contextual features.

## Main features

- CSV point-cloud loading with cluster-based train/test splitting.
- CloudCompare geometric and photometric features.
- KNN contextual feature enhancement.
- Spatially grouped cross-validation using 3D tiles.
- Interchangeable classifiers:
  - **XGBoost**
  - **Random Forest**
  - **Extra Trees**
- Hyperparameter search with model-specific parameter spaces.
- Accuracy, precision, recall, macro F1 and confusion matrix.
- Model export with `joblib`.
- Point-cloud classification colors compatible with Open3D workflows.
- Clear extension point for future models such as LightGBM or SVM.

## Project structure

```text
pointCloudClassification/
├── main_2.ipynb          # Main workflow
├── config.py             # Paths, model selection and project settings
├── data_utils.py         # Point-cloud loading and splitting
├── features.py           # KNN/contextual feature engineering
├── models.py             # Model registry and hyperparameter spaces
├── evaluation.py         # Training, CV and metrics
├── visualization.py      # Classified point-cloud visualization helpers
├── data/                 # Local datasets (not tracked by Git)
├── results/              # Trained models and metrics
├── outputs/              # Exported classified point clouds
├── requirements.txt
└── .gitignore
```

## Quick start

1. Create a Python environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Place the point-cloud CSV files in the paths defined in `config.py`.
4. Open `main_2.ipynb`.
5. Select the classifier in `config.py`:

```python
MODEL_NAME = "xgboost"
# or "random_forest"
# or "extra_trees"
```

6. Run the notebook from top to bottom.

The selected model uses its own hyperparameter space defined in `models.py`, so switching between the three classifiers does not require changing the training workflow.

## Adding another model

New classifiers can be added in `models.py` by registering the estimator and its hyperparameter search space. The rest of the workflow can then use it through the same `MODEL_NAME` mechanism.

## Notes

The original `main_2` workflow was used as the reference for the cleaned implementation. 

Large datasets, trained models and generated outputs should normally remain outside Git or be managed with Git LFS if they need to be versioned.
