import os
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained credit scoring model and save metrics to outputs folder.
    """

    print("\nEvaluating Credit Scoring Model...")

    # ============================
    # Predictions
    # ============================
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ============================
    # Metrics
    # ============================
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(cm)

    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # ============================
    # Save Metrics to File
    # ============================
    os.makedirs("outputs", exist_ok=True)
    metrics_path = "outputs/metrics.txt"

    with open(metrics_path, "w") as f:
        f.write("CREDIT SCORING MODEL EVALUATION\n")
        f.write("=" * 40 + "\n\n")

        f.write("Classification Report:\n")
        f.write(report + "\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")

        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")

    print(f"\nEvaluation metrics saved to: {metrics_path}")

    # ============================
    # ROC Curve Data (for plotting later if needed)
    # ============================
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    roc_data = {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }

    joblib.dump(roc_data, "outputs/roc_curve_data.pkl")
    print("ROC curve data saved to outputs/roc_curve_data.pkl")

    return {
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm
    }
