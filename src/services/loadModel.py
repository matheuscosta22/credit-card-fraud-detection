from pathlib import Path
import os
from huggingface_hub import HfApi
import joblib


MODEL_REPO = os.getenv("CCFD_HF_REPO")

def get_hf_api():
    """Retorna uma instância do HfApi. Separado para facilitar mocking em testes."""
    token = os.getenv("CCFD_HF_TOKEN")
    return HfApi(token=token)

def load_models():
    """Carrega modelos do diretório local ou baixa do HF Hub"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Buscar modelos no repositório
        api = get_hf_api()
        files = sorted([f for f in api.list_repo_files(MODEL_REPO) if f.endswith(".pkl")])
        
        # Encontrar último modelo
        model_file = None
        scaler_file = None
        
        for f in reversed(files):
            if "xgboost_model_" in f and not model_file:
                model_file = f
            elif "scaler_" in f and not scaler_file:
                scaler_file = f
            
            if model_file and scaler_file:
                break
        
        if not model_file or not scaler_file:
            raise RuntimeError("Modelo ou scaler não encontrado")
        
        # Baixar se não existir
        model_path = models_dir / model_file
        scaler_path = models_dir / scaler_file
        
        if not model_path.exists():
            api.hf_hub_download(repo_id=MODEL_REPO, filename=model_file, local_dir=models_dir, repo_type="model")
        if not scaler_path.exists():
            api.hf_hub_download(repo_id=MODEL_REPO, filename=scaler_file, local_dir=models_dir, repo_type="model")
        
        return joblib.load(model_path), joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")