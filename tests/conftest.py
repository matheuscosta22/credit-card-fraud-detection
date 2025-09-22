import os
import sys
import pytest
import pandas as pd
import json
from pathlib import Path

# Adicionar o diretório raiz do projeto ao PYTHONPATH
project_root = str(Path(__file__).parent.parent)
tests_dir = Path(__file__).parent
sys.path.insert(0, project_root)

@pytest.fixture(scope="session", autouse=True)
def setup_test_data(tmp_path_factory):
    """Criar dados de teste necessários"""
    # Criar diretório de dados dentro de tests
    data_dir = tests_dir / "data"
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Criar arquivo creditcard.csv para testes
    test_data = pd.DataFrame({
        'Time': [0, 3600, 7200],
        'Amount': [100.0, 50.0, 25.0],
        'V1': [0.1, 0.2, 0.3],
        'V2': [-0.1, -0.2, -0.3],
        'Class': [0, 1, 0]
    })
    
    test_data.to_csv(data_dir / "creditcard.csv", index=False)
    
    # Criar non_fraud_transactions.json
    non_fraud_data = []
    for i in range(200):  # Criar 200 transações de teste
        transaction = {
            'Time': i * 3600,
            'Amount': 100.0 / (i + 1),
            'V1': 0.1 * i,
            'V2': -0.1 * i,
        }
        # Adicionar V3 até V28
        for j in range(3, 29):
            transaction[f'V{j}'] = 0.1 * (i % 10)
        non_fraud_data.append(transaction)
    
    with open(data_dir / "non_fraud_transactions.json", "w") as f:
        json.dump(non_fraud_data, f)
    
    yield data_dir
    
    # Limpar após os testes
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
