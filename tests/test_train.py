import pytest
import pandas as pd
import numpy as np
from src.train import addFeatures, make_model
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

@pytest.fixture
def sample_transaction_data():
    return pd.DataFrame({
        'Time': [0, 3600, 7200, 10800, 14400, 18000],
        'Amount': [100.0, 50.0, 25.0, 75.0, 200.0, 150.0],
        'V1': [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
        'V2': [-0.1, -0.2, -0.3, -0.15, -0.25, -0.35],
        'Class': [0, 1, 0, 1, 0, 1]
    })

def test_add_features():
    # Criar dados de teste
    testData = pd.DataFrame({
        "Time": [0, 3600, 7200],
        "Amount": [100.0, 50.0, 25.0],
        "V1": [0.0, 0.1, 0.2]
    })
    
    # Executar função
    resultData = addFeatures(testData)
    
    # Verificar resultados
    assert 'log_amount' in resultData.columns
    assert 'amount_bin' in resultData.columns
    assert 'hour' in resultData.columns
    assert 'day' in resultData.columns
    
    # Verificar cálculos
    assert np.allclose(resultData['log_amount'], np.log1p(testData['Amount']))
    assert all(resultData['hour'] == [0, 1, 2])
    assert all(resultData['day'] == [0, 0, 0])

@pytest.fixture(autouse=True)
def mock_hf_api():
    """Mock para evitar upload real"""
    with patch('src.train.HfApi') as mock_api:
        mock_api.return_value.upload_file.return_value = None
        yield mock_api

def test_model_training(tmp_path, sample_transaction_data):
    """Testa o treinamento do modelo sem fazer upload real"""
    # Preparar diretórios
    dataDir = tmp_path / "data"
    metricsDir = tmp_path / "metrics"
    os.makedirs(dataDir)
    os.makedirs(metricsDir)
    
    # Salvar dados de teste
    sample_transaction_data.to_csv(dataDir / "creditcard.csv", index=False)
    
    # Treinar modelo sem salvar arquivos reais
    with patch('joblib.dump') as mock_dump, \
         patch('src.services.loadModel.load_models', return_value=(MagicMock(), MagicMock())):
        
        # Treinar
        make_model(dataDir)
        
        # Verificar salvamento local
        assert mock_dump.call_count == 4, "Modelo e scaler não foram salvos"
        
        # Verificar métricas
        metrics_file = metricsDir / "metrics.json"
        assert metrics_file.exists(), "Arquivo de métricas não foi criado"
        
        # Carregar e validar métricas
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        assert isinstance(metrics, dict), "Métricas inválidas"
        assert "accuracy" in metrics, "Métrica accuracy ausente"
        
        # Verificar métricas por classe
        for class_label in ["0", "1"]:
            class_metrics = metrics[class_label]
            for metric in ["precision", "recall", "f1-score", "support"]:
                assert metric in class_metrics, f"Métrica {metric} ausente para classe {class_label}"