"""
Feature Engineering Module for Credit Scoring Model
---------------------------------------------------
This script performs:
1. Data loading
2. Missing value handling
3. Categorical encoding
4. Feature scaling
5. New feature creation
6. Saving processed dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """Load raw dataset"""
    print("[INFO] Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def handle_missing_values(df):
    """Handle missing values"""
    print("[INFO] Handling missing values...")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_categorical_features(df):
    """Encode categorical variables using Label Encoding"""
    print("[INFO] Encoding categorical features...")

    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def create_new_features(df):
    """
    Create domain-specific features for credit scoring
    These features improve model learning
    """
    print("[INFO] Creating new features...")

    # Example common credit scoring features
    if "income" in df.columns and "loan_amount" in df.columns:
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)

    if "credit_limit" in df.columns and "current_balance" in df.columns:
        df["credit_utilization_ratio"] = df["current_balance"] / (df["credit_limit"] + 1)

    if "num_of_loans" in df.columns and "num_of_credit_cards" in df.columns:
        df["total_credit_accounts"] = df["num_of_loans"] + df["num_of_credit_cards"]

    if "age" in df.columns:
        df["age_bucket"] = pd.cut(
            df["age"],
            bins=[18, 25, 35, 45, 55, 65, 100],
            labels=[1, 2, 3, 4, 5, 6]
        )

    return df


def scale_numerical_features(df, target_column):
    """Scale numerical features using StandardScaler"""
    print("[INFO] Scaling numerical features...")

    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler


def save_processed_data(df, output_path):
    """Save processed dataset"""
    print("[INFO] Saving processed dataset...")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to: {output_path}")


def main():
    # File paths
    RAW_DATA_PATH = "data/raw/credit_data.csv"
    PROCESSED_DATA_PATH = "data/processed/credit_data_processed.csv"

    TARGET_COLUMN = "default"  # Change if your target column name is different

    # Step 1: Load data
    df = load_data(RAW_DATA_PATH)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Encode categorical features
    df, label_encoders = encode_categorical_features(df)

    # Step 4: Create new engineered features
    df = create_new_features(df)

    # Step 5: Scale numerical features
    df, scaler = scale_numerical_features(df, TARGET_COLUMN)

    # Step 6: Save processed dataset
    save_processed_data(df, PROCESSED_DATA_PATH)

    print("[SUCCESS] Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
