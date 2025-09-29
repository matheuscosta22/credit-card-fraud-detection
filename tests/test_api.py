import pytest
from fastapi.testclient import TestClient
from src.api import app
import json
from pathlib import Path
import asyncio
import psutil
import time
import httpx
from httpx import ASGITransport

testClient = TestClient(app)
testsDir = Path(__file__).parent

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
    with open(testsDir / "data/non_fraud_transactions.json") as f:
        transactionList = list(json.load(f))
        return {"transactions": transactionList[:5]}  # Pegar apenas 5 transações para teste


@pytest.fixture
def large_batch():
    with open(testsDir / "data/non_fraud_transactions.json") as f:
        transactionList = list(json.load(f))
        # Duplicar transações para criar um lote maior
        largeTransactionList = transactionList * 20  # 20x mais transações
        return {"transactions": largeTransactionList}


def test_root():
    apiResponse = testClient.get("/api/v1/fraud-detection/")
    assert apiResponse.status_code == 200
    assert apiResponse.json() == {"status": "API is running"}


def test_predict_single(sample_transaction):
    apiResponse = testClient.post(
        "/api/v1/fraud-detection/predict",
        json=sample_transaction
    )
    assert apiResponse.status_code == 200
    
    # Verificar estrutura da resposta
    responseData = apiResponse.json()
    assert "fraud_probability" in responseData
    assert "prediction" in responseData
    
    # Verificar tipos e valores
    assert isinstance(responseData["fraud_probability"], float)
    assert isinstance(responseData["prediction"], int)
    assert 0 <= responseData["fraud_probability"] <= 1
    assert responseData["prediction"] in [0, 1]


def test_predict_batch(sample_batch):
    apiResponse = testClient.post(
        "/api/v1/fraud-detection/predict/batch",
        json=sample_batch
    )
    assert apiResponse.status_code == 200
    
    # Como é streaming, precisamos ler linha por linha
    responseLines = apiResponse.iter_lines()
    partialResults = [json.loads(line) for line in responseLines]
    
    # Verificar último resultado
    lastResult = partialResults[-1]
    assert "partial_results" in lastResult
    resultData = lastResult["partial_results"]
    
    assert resultData["processed_count"] == len(sample_batch["transactions"])
    assert resultData["total_count"] == len(sample_batch["transactions"])
    assert "fraud_count" in resultData
    assert "non_fraud_count" in resultData
    
    # Verificar resultados individuais
    for resultItem in resultData["results"]:
        assert "index" in resultItem
        assert "fraud_probability" in resultItem
        assert "prediction" in resultItem
        assert isinstance(resultItem["fraud_probability"], float)
        assert isinstance(resultItem["prediction"], int)
        assert 0 <= resultItem["fraud_probability"] <= 1
        assert resultItem["prediction"] in [0, 1]


def test_large_batch(large_batch):
    startTime = time.time()
    processInfo = psutil.Process()
    initialMemory = processInfo.memory_info().rss / 1024 / 1024  # MB
    
    apiResponse = testClient.post(
        "/api/v1/fraud-detection/predict/batch",
        json=large_batch
    )
    assert apiResponse.status_code == 200
    
    # Ler e processar o streaming
    responseLines = apiResponse.iter_lines()
    partialResults = []
    processedIndices = set()
    
    for responseLine in responseLines:
        resultData = json.loads(responseLine)
        partialResults.append(resultData)
        
        # Verificar índices únicos
        batchData = resultData["partial_results"]
        for resultItem in batchData["results"]:
            assert resultItem["index"] not in processedIndices
            processedIndices.add(resultItem["index"])

    
    # Verificar completude
    lastResult = partialResults[-1]["partial_results"]
    assert lastResult["processed_count"] == len(large_batch["transactions"])
    assert len(processedIndices) == len(large_batch["transactions"])


    totalTime = time.time() - startTime
    finalMemory = processInfo.memory_info().rss / 1024 / 1024
    memoryUsage = finalMemory - initialMemory
    
    assert totalTime < 30  # Não deve demorar mais que 30 segundos
    assert memoryUsage < 1024  # Não deve usar mais que 1GB de memória adicional


def test_invalid_transaction():
    invalidData = {
        "Time": "invalid",  # deveria ser int
        "Amount": 100.0,
        "V1": 0.1
    }
    
    apiResponse = testClient.post(
        "/api/v1/fraud-detection/predict",
        json=invalidData
    )
    assert apiResponse.status_code == 422  # Erro de validação


def test_invalid_batch():
    invalidData = {
        "transactions": [
            {"Time": "invalid", "Amount": 100.0}  # transação inválida
        ]
    }
    
    apiResponse = testClient.post(
        "/api/v1/fraud-detection/predict/batch",
        json=invalidData
    )
    assert apiResponse.status_code == 422  # Erro de validação


def test_batch_concurrency(large_batch):

    async def makeRequest(httpClient: httpx.AsyncClient):
        apiResponse = await httpClient.post(
            "http://testserver/api/v1/fraud-detection/predict/batch",
            json=large_batch
        )
        return apiResponse.status_code


    async def executeRequests():
        transportHandler = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transportHandler, base_url="http://testserver") as httpClient:
            requestTasks = [makeRequest(httpClient) for _ in range(4)]  # Tentar 12 requisições simultâneas
            statusCodes = await asyncio.gather(*requestTasks)
            return statusCodes


    responseResults = asyncio.run(executeRequests())
    
    assert any(status == 200 for status in responseResults)  # Algumas devem ter sucesso
    assert any(status == 500 for status in responseResults)  # Algumas devem ser rejeitadas
