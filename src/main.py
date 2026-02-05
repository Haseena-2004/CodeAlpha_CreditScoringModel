import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import preprocess_and_engineer_features
from train_model import train_best_model
from evaluate_model import evaluate_model


def main():
    print("=" * 60)
    print("CREDIT SCORING MODEL - FULL PIPELINE STARTED")
    print("=" * 60)

    # ============================
    # 1. Load Dataset
    # ============================
    data_path = "data/credit_data.csv"
    print(f"\nLoading dataset from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print("Columns:", list(df.columns))

    # ============================
    # 2. Feature Engineering
    # ============================
    print("\nRunning feature engineering and preprocessing...")

    X, y = preprocess_and_engineer_features(df)

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # ============================
    # 3. Train-Test Split
    # ============================
    print("\nSplitting into train and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # ============================
    # 4. Train Model
    # ============================
    print("\nTraining and tuning model...")

    model = train_best_model(X_train, y_train)

    # ============================
    # 5. Evaluate Model
    # ============================
    print("\nEvaluating trained model...")

    results = evaluate_model(model, X_test, y_test)

    # ============================
    # 6. Final Summary
    # ============================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Final ROC-AUC Score: {results['roc_auc']:.4f}")
    print("All outputs saved in /outputs folder")
    print("Best model saved in /models folder")


if __name__ == "__main__":
    main()
