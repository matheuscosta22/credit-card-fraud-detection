import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from xgboost import XGBClassifier
import numpy as np
import json
from pathlib import Path

def convert_numpy(obj)-> dict:
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    # Transformação de Amount
    data["log_amount"] = np.log1p(data["Amount"])
    data["amount_bin"] = pd.qcut(data["Amount"], q=10, labels=False, duplicates="drop")

    # Features temporais a partir de Time
    data["hour"] = (data["Time"] // 3600) % 24
    data["day"] = data["Time"] // (3600 * 24)

    return data

def make_model(data_dir: str | Path = "data"):
    # Garantir diretórios
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # Carregar os dados
    data = pd.read_csv(Path(data_dir) / "creditcard.csv")

    # Criar novas features
    data = add_features(data)

    # Separar features e target
    features = data.drop(columns=["Class"])
    target = data["Class"]

    # Split em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criar modelo XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        eval_metric="logloss"
    )

    # Treinar modelo
    model.fit(X_train_scaled, y_train)

    # Previsões usando threshold fixo
    fraud_probs = model.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.2
    predictions = (fraud_probs >= threshold).astype(int)

    # Avaliar modelo com tratamento para divisão por zero
    report = classification_report(
        y_test, 
        predictions, 
        output_dict=True,
        zero_division=1  # Usar 1.0 em vez de 0.0 para métricas indefinidas
    )
    report_clean = convert_numpy(report)

    # Mostrar relatório
    print(json.dumps(report_clean, indent=4))

    # Salvar modelo e scaler
    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Salvar métricas em JSON
    with open("metrics/metrics.json", "w") as f:
        json.dump(report_clean, f, indent=4)

if __name__ == "__main__":
    make_model()
