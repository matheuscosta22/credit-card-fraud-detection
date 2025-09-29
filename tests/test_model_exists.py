import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.services.loadModel import load_models

@pytest.fixture
def mock_hf_api():
    """Mock para API do Hugging Face via get_hf_api()"""
    with patch('src.services.loadModel.get_hf_api') as mock_get_api:
        api = MagicMock()
        api.list_repo_files.return_value = [
            "xgboost_model_20250929_153254.pkl",
            "scaler_20250929_153254.pkl"
        ]
        mock_get_api.return_value = api
        yield api

def test_load_models_download_failure(mock_hf_api, tmp_path):
    """Testa comportamento quando falha o download do modelo"""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        (tmp_path / "models").mkdir(exist_ok=True)
        # Simular erro no download
        mock_hf_api.hf_hub_download.side_effect = Exception("Connection error")
        with pytest.raises(RuntimeError):
            load_models()
    finally:
        os.chdir(cwd)

def test_load_models_when_missing(mock_hf_api, tmp_path):
    """Testa se o modelo é baixado quando não existe localmente"""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Configurar mock do HF API para criar arquivos baixados
        def mock_download(**kwargs):
            filename = kwargs.get('filename')
            local_dir = kwargs.get('local_dir')
            file_path = Path(local_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()
            return str(file_path)
        mock_hf_api.hf_hub_download.side_effect = mock_download
        
        with patch('joblib.load', return_value=MagicMock()):
            load_models()
        
        # Verificar que baixou ambos
        assert mock_hf_api.hf_hub_download.call_count == 2
    finally:
        os.chdir(cwd)

def test_load_models_use_existing(mock_hf_api, tmp_path):
    """Testa se usa arquivos existentes sem baixar novamente"""
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        (models_dir / "xgboost_model_20250929_153254.pkl").touch()
        (models_dir / "scaler_20250929_153254.pkl").touch()
        
        with patch('joblib.load', return_value=MagicMock()):
            load_models()
        
        mock_hf_api.hf_hub_download.assert_not_called()
    finally:
        os.chdir(cwd)
