import os
import joblib
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
from ml import model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from ml.scripts.preprocess import preprocess_input_data, preprocess_training_data

# ---------------- Constants ---------------- #
MODELS_PATH = "ml/models/"
PLOTS_PATH = "backend/data/user_uploaded"
TRAIN_FILE_PATH = os.path.join(PLOTS_PATH, "train_data.csv")

# ---------------- MongoDB Setup ---------------- #
client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]
predictions_collection = db["predictions"]
summary_collection = db["summary"]

# ---------------- Ensure Directories ---------------- #
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# ---------------- Model Dictionary ---------------- #
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# ---------------- Auto Train If File Exists ---------------- #
def auto_train_if_file_exists(train_file_path=TRAIN_FILE_PATH, label_column="label"):
    if os.path.exists(train_file_path):
        print(f"[INFO] Training data found: {train_file_path}")
        df = preprocess_training_data(train_file_path, label_column)
        return train_models(df, label_column)
    else:
        raise FileNotFoundError(f"[ERROR] No training file at {train_file_path}")

# ---------------- Train Models ---------------- #
def train_models(df, label_column="label"):
    print("🚀 Training initiated...")
    X = df.drop(columns=[label_column])
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        try:
            print(f"🔧 Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = classification_report(y_test, y_pred, output_dict=True)

            model_file = os.path.join(MODELS_PATH, f"{name}.pkl")
            joblib.dump(model, model_file)
            print(f"✅ {name} model saved to {model_file}")

        except Exception as e:
            print(f"[ERROR] Failed to train {name}: {e}")

    print("✅ All models trained successfully.")
    return results

# ---------------- Predict and Generate Outputs ---------------- #
def predict_and_generate_outputs(filepath):
    print(f"📂 Loading input file: {filepath}")
    df = preprocess_input_data(filepath)
    full_predictions = df.copy()

    predictions = {}
    probabilities = {}

    for name in models:
        model_file = os.path.join(MODELS_PATH, f"{name}.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"[ERROR] Model not found: {model_file}")

        print(f"🔍 Loading model: {name}")
        model = joblib.load(model_file)

        try:
            preds = model.predict(df)
            full_predictions[f"prediction_{name}"] = preds
            predictions[name] = preds

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)[:, 1]
                full_predictions[f"probability_{name}"] = probs
                probabilities[name] = probs
            else:
                full_predictions[f"probability_{name}"] = np.nan
                probabilities[name] = [None] * len(df)

        except Exception as e:
            print(f"[ERROR] Prediction error in {name}: {e}")
            full_predictions[f"prediction_{name}"] = np.nan
            full_predictions[f"probability_{name}"] = np.nan
            predictions[name] = []
            probabilities[name] = []

    # 🧠 Majority Voting
    print("📊 Performing majority vote...")
    pred_cols = [col for col in full_predictions.columns if col.startswith("prediction_")]
    full_predictions["majority_vote"] = full_predictions[pred_cols].mode(axis=1)[0]

    def models_voted_fraud(row):
        return ", ".join(
            col.replace("prediction_", "")
            for col in pred_cols if row[col] == "fraud"
        ) or "None"

    full_predictions["used_models"] = full_predictions.apply(models_voted_fraud, axis=1)

    # ⚠️ Check transaction_amount existence
    if "transaction_amount" not in full_predictions.columns:
        print("⚠️ Warning: 'transaction_amount' column missing!")

    # 📈 Summary by user
    print("📈 Generating user summary...")
    user_summary = (
        full_predictions.groupby("user_id")["majority_vote"]
        .value_counts().unstack(fill_value=0).reset_index()
    )

    # 📊 Charts
    charts_data = {
        "overall_distribution": full_predictions["majority_vote"].value_counts().to_dict(),
        "fraud_by_type": full_predictions[full_predictions["majority_vote"] == "fraud"]["transaction_type"]
            .value_counts().to_dict(),
        "user_flags": full_predictions.groupby("user_id")["majority_vote"]
            .apply(lambda x: (x == "fraud").sum()).to_dict()
    }

    # 🌡️ Heatmap
    print("🌡️ Creating heatmap...")
    heatmap_data = pd.crosstab(full_predictions["user_id"], full_predictions["majority_vote"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
    plt.title("Fraud Detection Heatmap by User")
    heatmap_file = os.path.join(PLOTS_PATH, "heatmap.png")
    plt.savefig(heatmap_file)
    plt.close()
    print(f"✅ Heatmap saved: {heatmap_file}")

    # 🗃️ MongoDB Storage
    print("📦 Storing results to MongoDB...")
    timestamp = datetime.datetime.now().isoformat()

    full_predictions["timestamp"] = timestamp
    predictions_collection.insert_many(full_predictions.to_dict(orient="records"))

    user_summary["timestamp"] = timestamp
    summary_collection.insert_many(user_summary.to_dict(orient="records"))

    return full_predictions, predictions, {
        "user_summary": user_summary,
        "charts_data": charts_data,
        "heatmap": os.path.basename(heatmap_file)
    }
