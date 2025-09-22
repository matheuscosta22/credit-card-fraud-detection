import pytest
from fastapi.testclient import TestClient
from src.api import app
import json
from pathlib import Path

client = TestClient(app)
tests_dir = Path(__file__).parent

@pytest.fixture
def sample_transaction():
    return {
        "Time": 0,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }

@pytest.fixture
def sample_batch():
    with open(tests_dir / "data/non_fraud_transactions.json") as f:
        transactions = list(json.load(f))
        return {"transactions": transactions[:5]}  # Pegar apenas 5 transações para teste

def test_root():
    response = client.get("/api/v1/fraud-detection/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

def test_predict_single(sample_transaction):
    response = client.post(
        "/api/v1/fraud-detection/predict",
        json=sample_transaction
    )
    assert response.status_code == 200
    
    # Verificar estrutura da resposta
    data = response.json()
    assert "fraud_probability" in data
    assert "prediction" in data
    
    # Verificar tipos e valores
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["prediction"], int)
    assert 0 <= data["fraud_probability"] <= 1
    assert data["prediction"] in [0, 1]

def test_predict_batch(sample_batch):
    response = client.post(
        "/api/v1/fraud-detection/predict/batch",
        json=sample_batch
    )
    assert response.status_code == 200
    
    # Verificar estrutura da resposta
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == len(sample_batch["transactions"])
    
    # Verificar cada resultado
    for result in data["results"]:
        assert "index" in result
        assert "fraud_probability" in result
        assert "prediction" in result
        assert isinstance(result["fraud_probability"], float)
        assert isinstance(result["prediction"], int)
        assert 0 <= result["fraud_probability"] <= 1
        assert result["prediction"] in [0, 1]

def test_invalid_transaction():
    invalid_transaction = {
        "Time": "invalid",  # deveria ser int
        "Amount": 100.0,
        "V1": 0.1
    }
    
    response = client.post(
        "/api/v1/fraud-detection/predict",
        json=invalid_transaction
    )
    assert response.status_code == 422  # Erro de validação

def test_invalid_batch():
    invalid_batch = {
        "transactions": [
            {"Time": "invalid", "Amount": 100.0}  # transação inválida
        ]
    }
    
    response = client.post(
        "/api/v1/fraud-detection/predict/batch",
        json=invalid_batch
    )
    assert response.status_code == 422  # Erro de validação
