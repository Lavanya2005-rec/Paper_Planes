# ml_utils.py

import joblib
import datetime
import pandas as pd

def load_model_and_encoder():
    model = joblib.load("model/fraud_model.pkl")
    encoder = joblib.load("model/label_encoder.pkl")
    return model, encoder

def make_predictions(df: pd.DataFrame, model, encoder):
    preds = model.predict(df)
    probs = model.predict_proba(df).max(axis=1)
    df["transaction_type"] = encoder.inverse_transform(preds)
    df["risk_score"] = (probs * 100).round(2)
    df["predicted_at"] = datetime.datetime.utcnow()
    return df
