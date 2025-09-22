import pytest
import time
import json
import numpy as np
from src.services.fraudDetection import predict, predict_batch
from src.types.transaction import Transaction
from pathlib import Path

tests_dir = Path(__file__).parent

@pytest.fixture
def sample_transaction():
    return Transaction(
        Time=0,
        V1=-1.3598071336738,
        V2=-0.0727811733098497,
        V3=2.53634673796914,
        V4=1.37815522427443,
        V5=-0.338320769942518,
        V6=0.462387777762292,
        V7=0.239598554061257,
        V8=0.0986979012610507,
        V9=0.363786969611213,
        V10=0.0907941719789316,
        V11=-0.551599533260813,
        V12=-0.617800855762348,
        V13=-0.991389847235408,
        V14=-0.311169353699879,
        V15=1.46817697209427,
        V16=-0.470400525259478,
        V17=0.207971241929242,
        V18=0.0257905801985591,
        V19=0.403992960255733,
        V20=0.251412098239705,
        V21=-0.018306777944153,
        V22=0.277837575558899,
        V23=-0.110473910188767,
        V24=0.0669280749146731,
        V25=0.128539358273528,
        V26=-0.189114843888824,
        V27=0.133558376740387,
        V28=-0.0210530534538215,
        Amount=149.62
    )

def test_single_prediction_performance(sample_transaction):
    # Medir tempo de uma única predição
    start_time = time.time()
    predict(sample_transaction)
    end_time = time.time()
    
    prediction_time = end_time - start_time
    assert prediction_time < 0.1, f"Predição muito lenta: {prediction_time:.4f}s"

def test_batch_prediction_latency():
    # Carregar transações de teste
    with open(tests_dir / "data/non_fraud_transactions.json") as f:
        transactions_data = list(json.load(f))
        transactions = [Transaction(**t) for t in transactions_data[:100]]  # Testar com 100 transações
    
    # Medir tempo de resposta
    start_time = time.time()
    results = predict_batch(transactions)
    end_time = time.time()
    
    # Verificar latência média por transação
    total_time = end_time - start_time
    latency_per_transaction = total_time / len(transactions)
    
    # A latência por transação deve ser menor que 10ms
    assert latency_per_transaction < 0.01, f"Latência muito alta: {latency_per_transaction:.4f}s por transação"

def test_batch_size_scaling():
    # Testar diferentes tamanhos de batch
    batch_sizes = [10, 50, 100]
    latencies = []
    
    with open(tests_dir / "data/non_fraud_transactions.json") as f:
        all_transactions_data = list(json.load(f))
    
    for size in batch_sizes:
        transactions = [Transaction(**t) for t in all_transactions_data[:size]]
        
        # Medir tempo para cada tamanho de batch
        start_time = time.time()
        predict_batch(transactions)
        end_time = time.time()
        
        latencies.append((end_time - start_time) / size)
    
    # Verificar se a latência por transação se mantém estável
    latency_variation = np.std(latencies)
    assert latency_variation < 0.005, f"Variação de latência muito alta: {latency_variation:.4f}s"

def test_memory_usage(sample_transaction):
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Fazer 1000 predições
    for _ in range(1000):
        predict(sample_transaction)
    
    final_memory = process.memory_info().rss
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    
    # O aumento de memória deve ser menor que 100MB
    assert memory_increase < 100, f"Uso de memória muito alto: {memory_increase:.2f}MB"
