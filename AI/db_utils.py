# db_utils.py

from pymongo import MongoClient
import pandas as pd
from config import MONGO_URI

client = MongoClient(MONGO_URI)
db = client.fraudshield
total_collection = db.all_transactions
userwise_collection = db.userwise_transactions

def store_all_transactions(df: pd.DataFrame):
    records = df.to_dict(orient="records")
    if records:
        total_collection.insert_many(records)

def store_userwise_transactions(df: pd.DataFrame):
    for user_id, group_df in df.groupby("user_id"):
        records = group_df.to_dict(orient="records")
        userwise_collection.update_one(
            {"user_id": user_id},
            {"$push": {"transactions": {"$each": records}}},
            upsert=True
        )
