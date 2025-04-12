import pandas as pd
import json
import os
import numpy as np
import time

FEATURE_STORE_PATH = "ml/models/features.json"
TRAIN_UPLOAD_PATH = "backend/data/user_uploaded/train"
PREDICT_UPLOAD_PATH = "backend/data/user_uploaded/predict"
PROCESSED_PATH = "backend/data/processed"
LABEL_COLUMN = "label"  # Change this to your actual label column


# Save feature metadata during training
def save_feature_metadata(df, label_column):
    metadata = {
        "features": df.drop(columns=[label_column]).columns.tolist(),
        "label": label_column
    }
    with open(FEATURE_STORE_PATH, "w") as f:
        json.dump(metadata, f)
    print("[INFO] Feature structure saved to 'features.json'.")


# Training data preprocessing
def preprocess_training_data(filepath, label_column):
    df = pd.read_csv(filepath)

    if label_column not in df.columns:
        raise ValueError(f"[ERROR] Label column '{label_column}' is missing in training data.")

    df.fillna("Unknown", inplace=True)

    # Identify categorical columns (excluding label)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if label_column in categorical_cols:
        categorical_cols.remove(label_column)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save the structure of the features
    save_feature_metadata(df, label_column)

    return df


# Prediction data preprocessing
def preprocess_input_data(filepath):
    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError("[ERROR] Feature metadata not found. Train the model first.")

    # Load feature metadata
    with open(FEATURE_STORE_PATH, "r") as f:
        metadata = json.load(f)

    expected_features = metadata["features"]
    label_column = metadata["label"]

    # Load raw file
    df = pd.read_csv(filepath)

    # Preserve original columns
    original_columns = ['transaction_id', 'user_id', 'username', 'transaction_amount']
    preserved = df[original_columns].copy() if all(col in df.columns for col in original_columns) else pd.DataFrame()

    # Drop label if exists
    if label_column in df.columns:
        df.drop(columns=[label_column], inplace=True)

    # Fill missing values
    df.fillna("Unknown", inplace=True)

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Add any missing features
    for col in expected_features:
        if col not in df.columns:
            df[col] = 1 if "_Unknown" in col else 0

    # Ensure feature order
    df = df[expected_features]

    # Merge preserved info back for post-prediction usage
    if not preserved.empty:
        df = pd.concat([preserved.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    return df


# Auto-detect and preprocess latest file
def _auto_preprocess(upload_path, output_filename, preprocess_fn, *args):
    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    uploaded_files = [f for f in os.listdir(upload_path) if f.endswith('.csv')]
    if not uploaded_files:
        print(f"[INFO] No file uploaded yet in '{upload_path}'.")
        return None

    # Pick latest uploaded CSV
    latest_file = max(uploaded_files, key=lambda x: os.path.getctime(os.path.join(upload_path, x)))
    full_path = os.path.join(upload_path, latest_file)

    try:
        df = preprocess_fn(full_path, *args)
        output_path = os.path.join(PROCESSED_PATH, output_filename)
        df.to_csv(output_path, index=False)
        print(f"[INFO] File '{latest_file}' processed and saved to '{output_path}'.")
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to preprocess '{latest_file}': {str(e)}")
        return None


# Wrapper functions for each mode
def preprocess_latest_train_file():
    return _auto_preprocess(TRAIN_UPLOAD_PATH, "preprocessed_train.csv", preprocess_training_data, LABEL_COLUMN)


def preprocess_latest_predict_file():
    return _auto_preprocess(PREDICT_UPLOAD_PATH, "preprocessed_predict.csv", preprocess_input_data)


# Continuous preprocessing listener (optional)
if __name__ == "__main__":
    while True:
        preprocess_latest_train_file()
        preprocess_latest_predict_file()
        time.sleep(10)
