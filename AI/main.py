from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from joblib import load
from datetime import datetime
from io import StringIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["fraudshield"]
collection_all = db["all_transactions"]
collection_user = db["user_transactions"]

# Load ML model and label encoder
model = load('model/fraud_model.pkl')
label_encoder = load('model/label_encoder.pkl')

# Define required columns for model input
REQUIRED_COLUMNS = [
    'user_id', 'transaction_id', 'transaction_amount', 'location', 'ip_address',
    'device_type', 'vpn_status', 'mac_address', 'geo_lat', 'geo_long', 'kyc_status',
    'transaction_time', 'user_name', 'user_email', 'card_last4', 'device_id',
    'geo_anomaly', 'ip_risk', 'velocity_count', 'velocity_flag', 'past_avg_spend',
    'user_agent', 'hour', 'is_night', 'location_encoded',
    'risk_score', 'transaction_type_encoded'
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    # Read CSV
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df.fillna("unknown", inplace=True)

    # Preserve original transaction_time as string
    if 'transaction_time' in df.columns:
        df['transaction_time'] = df['transaction_time'].astype(str)

    # Add derived fields
    df['hour'] = pd.to_datetime(df['transaction_time'], errors='coerce').dt.hour.fillna(0).astype(int)
    df['email_domain'] = df['user_email'].astype(str).apply(lambda x: x.split('@')[-1] if '@' in x else 'unknown')

    # Add default values if columns missing
    if 'transaction_type_encoded' not in df.columns:
        df['transaction_type_encoded'] = 0
    if 'risk_score' not in df.columns:
        df['risk_score'] = 50.0

    # Final column check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

    # Convert or drop 'transaction_time' column to numeric or exclude it
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')  # Convert to datetime
    df['transaction_time'] = df['transaction_time'].astype('int64') // 10**9  # Convert to UNIX timestamp (seconds)
    
    # Encode object-type columns except specific preserved ones
    columns_to_exclude = {'transaction_time'}
    for col in df.columns:
        if df[col].dtype == 'object' and col not in columns_to_exclude:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Prediction
    model_input = df[REQUIRED_COLUMNS]
    predictions = model.predict(model_input)
    df['model_prediction'] = label_encoder.inverse_transform(predictions)

    # Custom risk score calculation (if needed)
    df['risk_score'] = (
        df['transaction_amount'].astype(float) * 0.3 +
        df['ip_risk'].astype(float) * 0.4 +
        df['geo_anomaly'].astype(float) * 0.3
    )

    # Add prediction timestamp
    df['predicted_at'] = datetime.now().isoformat()

    # Store in MongoDB (preserving original values)
    records = df.to_dict(orient="records")
    collection_all.insert_many(records)

    # Group by user_id for user-specific collection
    grouped = df.groupby('user_id')
    for user_id, user_df in grouped:
        collection_user.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id": user_id,
                "transactions": user_df.to_dict(orient='records')
            }},
            upsert=True
        )

    return {"message": "✅ File processed, predicted, and stored in MongoDB!"}
