from typing import List
from src.train import add_features
from src.types.transaction import Transaction
import pandas as pd
import joblib

model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict(transaction: Transaction) -> dict:
    threshold = 0.2

    data = add_features(
        pd.DataFrame([transaction.model_dump()])
    )

    features_scaled = scaler.transform(data)

    prob = model.predict_proba(features_scaled)[0][1]
    prediction = int(prob >= threshold)

    return {
        "fraud_probability": float(prob),
        "prediction": prediction
    }


def predict_batch(transactions: List[Transaction]) -> dict:
    data = pd.DataFrame([t.model_dump() for t in transactions])
    data = add_features(data)
    features_scaled = scaler.transform(data)

    threshold = 0.2
    probabilities = model.predict_proba(features_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = [
        {
            "index": i,
            "fraud_probability": float(probability),
            "prediction": int(prediction)
        }
        for i, (probability, prediction) in enumerate(zip(probabilities, predictions))
    ]

    return {
        "count_fraud": int(predictions.sum()),
        "count_non_fraud": int(len(predictions) - predictions.sum()),
        "results": results
    }