import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_best_model(X_train, y_train):
    """
    Train and tune a Random Forest classifier for credit scoring.
    Uses cross-validation and saves the best model to disk.
    """

    print("\n==============================")
    print("TRAINING CREDIT SCORING MODEL")
    print("==============================")

    # ============================
    # Base Model
    # ============================
    rf_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    # ============================
    # Stratified CV (important for credit risk)
    # ============================
    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # ============================
    # Hyperparameter Grid
    # ============================
    param_grid = {
        "n_estimators": [150, 250, 350],
        "max_depth": [None, 6, 12, 18],
        "min_samples_split": [2, 5, 8],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2"]
    }

    print("\nStarting GridSearchCV for hyperparameter tuning...")
    print("This may take some time depending on dataset size.\n")

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    # ============================
    # Model Training
    # ============================
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\n==============================")
    print("BEST MODEL SELECTED")
    print("==============================")

    print("Best Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")

    # ============================
    # Feature Importance (Optional but Professional)
    # ============================
    try:
        importances = best_model.feature_importances_
        print("\nTop Feature Importances:")
        top_idx = np.argsort(importances)[::-1][:10]
        for i in top_idx:
            print(f"  Feature {i} Importance: {importances[i]:.4f}")
    except Exception as e:
        print("Could not compute feature importances:", e)

    # ============================
    # Save Model
    # ============================
    model_path = "models/best_credit_model.pkl"
    joblib.dump(best_model, model_path)

    print("\n==============================")
    print("MODEL SAVED SUCCESSFULLY")
    print("==============================")
    print(f"Model path: {model_path}")

    return best_model
