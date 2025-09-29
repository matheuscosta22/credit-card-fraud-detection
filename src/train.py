import os
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
from datetime import datetime
from huggingface_hub import HfApi

def convert_numpy(obj) -> dict:
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def addFeatures(inputData: pd.DataFrame) -> pd.DataFrame:
    resultData = inputData.copy()
    resultData.loc[:, "log_amount"] = np.log1p(resultData["Amount"])
    resultData.loc[:, "amount_bin"] = pd.qcut(resultData["Amount"], q=10, labels=False, duplicates="drop")
    resultData.loc[:, "hour"] = (resultData["Time"] // 3600) % 24
    resultData.loc[:, "day"] = resultData["Time"] // (3600 * 24)
    return resultData

def make_model(dataDir: str | Path = "data", repo_id: str | None = None):
    dataDir = Path(dataDir)
    modelDir = dataDir.parent / "models"
    metricsDir = dataDir.parent / "metrics"

    os.makedirs(modelDir, exist_ok=True)
    os.makedirs(metricsDir, exist_ok=True)

    # Resolver repo_id a partir de variável de ambiente, se não informado
    if repo_id is None:
        repo_id = os.getenv("CCFD_HF_REPO")

    # Carregar dataset
    inputData = pd.read_csv(dataDir / "creditcard.csv")

    processedData = addFeatures(inputData)
    featureData = processedData.drop(columns=["Class"])
    targetData = processedData["Class"]

    # Split dataset
    trainFeatures, testFeatures, trainTarget, testTarget = train_test_split(
        featureData, targetData, test_size=0.2, random_state=42, stratify=targetData
    )

    # Normalização
    modelScaler = StandardScaler()
    trainFeaturesScaled = modelScaler.fit_transform(trainFeatures)
    testFeaturesScaled = modelScaler.transform(testFeatures)

    # Modelo XGBoost
    modelXgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(trainTarget[trainTarget == 0]) / len(trainTarget[trainTarget == 1]),
        random_state=42,
        eval_metric="logloss"
    )
    modelXgb.fit(trainFeaturesScaled, trainTarget)

    # Predições e métricas
    fraudPredictions = modelXgb.predict(testFeaturesScaled)
    modelReport = classification_report(testTarget, fraudPredictions, output_dict=True, zero_division=1)
    reportClean = convert_numpy(modelReport)
    print(json.dumps(reportClean, indent=4))

    # === Versionamento ===
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Caminhos locais
    model_path = modelDir / f"xgboost_model_{version_tag}.pkl"
    scaler_path = modelDir / f"scaler_{version_tag}.pkl"
    metrics_path = metricsDir / f"metrics_{version_tag}.json"

    # Salvar versões locais
    joblib.dump(modelXgb, model_path)
    joblib.dump(modelScaler, scaler_path)
    with open(metrics_path, "w") as f:
        json.dump(reportClean, f, indent=4)

    # Salvar versão principal (última)
    joblib.dump(modelXgb, modelDir / "xgboost_model.pkl")
    joblib.dump(modelScaler, modelDir / "scaler.pkl")
    with open(metricsDir / "metrics.json", "w") as f:
        json.dump(reportClean, f, indent=4)

    print(f"✅ Modelo salvo localmente em {model_path}")

    # === Upload para Hugging Face Hub ===
    try:
        hf_token = os.getenv("CCFD_HF_TOKEN")
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"xgboost_model_{version_tag}.pkl",
            repo_id=repo_id,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=str(scaler_path),
            path_in_repo=f"scaler_{version_tag}.pkl",
            repo_id=repo_id,
            repo_type="model"
        )
        api.upload_file(
            path_or_fileobj=str(metrics_path),
            path_in_repo=f"metrics_{version_tag}.json",
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ Arquivos enviados para o Hugging Face Hub em {repo_id}")
    except Exception as e:
        print(f"⚠️ Erro ao enviar para Hugging Face Hub: {e}")

if __name__ == "__main__":
    make_model()
