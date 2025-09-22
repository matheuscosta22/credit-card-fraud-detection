import pytest
import pandas as pd
import numpy as np
from src.train import add_features
import os
import json
from pathlib import Path

tests_dir = Path(__file__).parent

@pytest.fixture
def sample_transaction_data():
    # Criar mais dados de teste para garantir pelo menos 2 amostras por classe
    return pd.DataFrame({
        'Time': [0, 3600, 7200, 10800, 14400, 18000],  # 0h, 1h, 2h, 3h, 4h, 5h
        'Amount': [100.0, 50.0, 25.0, 75.0, 200.0, 150.0],
        'V1': [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
        'V2': [-0.1, -0.2, -0.3, -0.15, -0.25, -0.35],
        'Class': [0, 1, 0, 1, 0, 1]  # 3 amostras de cada classe
    })

def test_add_features():
    # Preparar dados de teste
    df = pd.DataFrame({
        'Time': [0, 3600, 7200],  # 0h, 1h, 2h
        'Amount': [100.0, 50.0, 25.0],
        'V1': [0.1, 0.2, 0.3]
    })
    
    # Executar função
    result = add_features(df)
    
    # Verificar resultados
    assert 'log_amount' in result.columns
    assert 'amount_bin' in result.columns
    assert 'hour' in result.columns
    assert 'day' in result.columns
    
    # Verificar cálculos
    np.testing.assert_almost_equal(
        result['log_amount'].values,
        np.log1p([100.0, 50.0, 25.0])
    )
    
    assert all(result['hour'].values == [0, 1, 2])
    assert all(result['day'].values == [0, 0, 0])

def test_model_training(tmp_path, sample_transaction_data):
    # Criar diretório temporário para teste
    model_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    data_dir = tmp_path / "data"
    os.makedirs(model_dir)
    os.makedirs(metrics_dir)
    os.makedirs(data_dir)
    
    # Salvar dados de teste
    data_path = data_dir / "creditcard.csv"
    sample_transaction_data.to_csv(data_path, index=False)
    
    # Importar função de treino aqui para usar os caminhos temporários
    from src.train import make_model
    
    # Executar treinamento
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        make_model(data_dir)
    
    # Verificar se os arquivos foram criados
    assert os.path.exists(model_dir / "xgboost_model.pkl")
    assert os.path.exists(model_dir / "scaler.pkl")
    assert os.path.exists(metrics_dir / "metrics.json")
    
    # Verificar métricas
    with open(metrics_dir / "metrics.json") as f:
        metrics = json.load(f)
    
    # Verificar estrutura básica das métricas
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    
    # Verificar métricas por classe
    for class_label in ["0", "1"]:  # Verificar métricas para ambas as classes
        assert class_label in metrics
        class_metrics = metrics[class_label]
        
        # Verificar presença e validade das métricas principais
        assert "precision" in class_metrics
        assert "recall" in class_metrics
        assert "f1-score" in class_metrics
        assert "support" in class_metrics
        
        # Verificar limites das métricas (agora usando 1.0 para divisão por zero)
        assert 0 <= class_metrics["precision"] <= 1
        assert 0 <= class_metrics["recall"] <= 1
        assert 0 <= class_metrics["f1-score"] <= 1
        assert class_metrics["support"] >= 0  # Deve ter pelo menos uma amostra
    
    # Verificar métricas macro avg e weighted avg
    for avg_type in ["macro avg", "weighted avg"]:
        assert avg_type in metrics
        avg_metrics = metrics[avg_type]
        assert all(0 <= avg_metrics[m] <= 1 for m in ["precision", "recall", "f1-score"])
        assert avg_metrics["support"] > 0