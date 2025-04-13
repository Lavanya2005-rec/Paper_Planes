import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from joblib import dump
from pymongo import MongoClient
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import numpy as np
import os

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Encode 'transaction_type' using LabelEncoder (0,1,2)
le = LabelEncoder()
df['transaction_type_encoded'] = le.fit_transform(df['transaction_type'])

# Create label mapping (convert np.int64 to plain int for MongoDB)
label_mapping = {label: int(code) for label, code in zip(le.classes_, le.transform(le.classes_))}
print("✅ Label Mapping:", label_mapping)

# Feature engineering
df['transaction_hour'] = pd.to_datetime(df['transaction_time'], errors='coerce').dt.hour
df['email_domain'] = df['user_email'].astype(str).apply(lambda x: x.split('@')[-1] if '@' in x else 'unknown')

# Handle missing values
df.fillna("unknown", inplace=True)

# Encode all object columns EXCEPT the original 'transaction_type'
for col in df.columns:
    if df[col].dtype == 'object' and col != 'transaction_type':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Prepare features and target
X = df.drop(['transaction_type'], axis=1)
y = df['transaction_type_encoded']

# Standardize features for AutoEncoder
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define supervised classifiers
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(silent=True),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

# Evaluate models using cross-validation and store accuracy scores
model_accuracies = {}
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    model_accuracies[name] = score
    trained_models[name] = model
    print(f"Accuracy for {name}: {score}")

# Train IsolationForest (unsupervised)
iso_model = IsolationForest(contamination=0.05, random_state=42)
iso_model.fit(X_train)
iso_pred = iso_model.predict(X_test)
iso_pred_classified = np.where(iso_pred == -1, 1, 0)  # -1 is anomaly
iso_score = accuracy_score(y_test == 1, iso_pred_classified)
model_accuracies['Isolation Forest'] = iso_score
trained_models['Isolation Forest'] = iso_model
print(f"Accuracy for Isolation Forest: {iso_score}")

# Train AutoEncoder (unsupervised)
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(16, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(8, activation="relu")(encoder)
decoder = Dense(16, activation='relu')(encoder)
decoder = Dense(input_dim, activation='linear')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)

# Predict and score
X_test_reconstructed = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_reconstructed, 2), axis=1)
threshold = np.percentile(mse, 95)
auto_pred_classified = (mse > threshold).astype(int)
auto_score = accuracy_score(y_test == 1, auto_pred_classified)
model_accuracies['AutoEncoder'] = auto_score
trained_models['AutoEncoder'] = autoencoder
print(f"Accuracy for AutoEncoder: {auto_score}")

# Sort and select top 5 models
top_5 = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n🔝 Top 5 Models:")
for name, score in top_5:
    print(f"{name}: {score}")

# Train the best model
best_model_name = top_5[0][0]
best_model = trained_models[best_model_name]
print(f"\nTraining the best model: {best_model_name}...")
if best_model_name not in ['AutoEncoder', 'Isolation Forest']:
    best_model.fit(X_train, y_train)
    df['model_prediction'] = best_model.predict(X_scaled)
else:
    if best_model_name == 'Isolation Forest':
        pred = best_model.predict(X_scaled)
        df['model_prediction'] = np.where(pred == -1, 1, 0)
    else:
        recon = best_model.predict(X_scaled)
        mse_all = np.mean(np.power(X_scaled - recon, 2), axis=1)
        threshold = np.percentile(mse_all, 95)
        df['model_prediction'] = (mse_all > threshold).astype(int)

# Save best model and encoder
os.makedirs("model", exist_ok=True)
dump(best_model, f'model/fraud_model_best.pkl')
dump(le, 'model/label_encoder.pkl')
dump(scaler, 'model/scaler.pkl')
print("✅ Best model, encoder, and scaler saved.")

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['fraudshield']

# Save the processed dataset
collection = db['trained_transactions']
collection.delete_many({})
collection.insert_many(df.to_dict(orient='records'))
print("✅ Processed dataset saved to MongoDB.")

# Save model metadata and top 5 scores
db['model_registry'].insert_one({
    "model_path": "model/fraud_model_best.pkl",
    "label_encoder_path": "model/label_encoder.pkl",
    "scaler_path": "model/scaler.pkl",
    "created_at": datetime.now().isoformat(),
    "label_mapping": label_mapping,
    "top_5_models": {k: float(v) for k, v in top_5},
})
print("✅ Top 5 model metadata saved to MongoDB.")
