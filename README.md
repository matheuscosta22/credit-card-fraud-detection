
# Detector de Fraudes em Cartão de Crédito

API para detecção de fraudes em transações de cartão de crédito usando modelo XGBoost. O modelo foi treinado para identificar transações suspeitas com base em 28 características anonimizadas (V1-V28), além do tempo e valor da transação.

## Recursos utilizados

- Python 3.11
- FastAPI
- Scikit-learn
- pandas
- XGBoost
- Poetry
- Docker compose

## Estrutura do Projeto

```
├── data/               # Dados para treino e predição
├── metrics/            # Métricas do modelo
├── models/             # Modelos treinados
├── src/
│   ├── services/       # Serviços de predição
│   ├── types/          # Tipos e schemas
│   ├── api.py          # API FastAPI
│   └── train.py        # Treinamento do modelo
├── tests/              # Testes automatizados
├── docker-compose.yml  # Configuração Docker
└── pyproject.toml     # Dependências Poetry
```


## Como executar projeto

Será necessário ter o docker e docker compose instalado no sistema.
Link da documentação: https://docs.docker.com/compose/install/

Após clonar o projeto, instalar o docker e docker compose, execute o seguinte comando:
```bash
docker compose up --build
```

Após o build, o projeto será executado e a API estará disponível no endereço http://localhost:8000

```bash
Attaching to fraud-api
fraud-api  | INFO:     Will watch for changes in these directories: ['/app']
fraud-api  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
fraud-api  | INFO:     Started reloader process [1] using StatReload
fraud-api  | INFO:     Started server process [13]
fraud-api  | INFO:     Waiting for application startup.
fraud-api  | INFO:     Application startup complete.
```


## Rotas

### Analisar Uma Transação

```bash
curl -X POST "http://localhost:8000/api/v1/fraud-detection/predict" \
-H "Content-Type: application/json" \
-d '{
    "Time": 0,
    "Amount": 100.0,
    "V1": 0.1,
    "V2": -0.1,
    "V3": 0.2,
    "V4": -0.2,
    "V5": 0.3,
    "V6": -0.3,
    "V7": 0.4,
    "V8": -0.4,
    "V9": 0.5,
    "V10": -0.5,
    "V11": 0.6,
    "V12": -0.6,
    "V13": 0.7,
    "V14": -0.7,
    "V15": 0.8,
    "V16": -0.8,
    "V17": 0.9,
    "V18": -0.9,
    "V19": 1.0,
    "V20": -1.0,
    "V21": 1.1,
    "V22": -1.1,
    "V23": 1.2,
    "V24": -1.2,
    "V25": 1.3,
    "V26": -1.3,
    "V27": 1.4,
    "V28": -1.4
}'
```


### Exemplo de Resposta

```json
{
    "fraud_probability": 0.15,  // Probabilidade de fraude (0 a 1)
    "prediction": 0           // 0: Normal, 1: Fraude
}
```


### Analisar Várias Transações

No diretório `data` existe um arquivo `fraud_transactions.json` que contém 200 transações fraudulentas e existe um arquivo `non_fraud_transactions.json` que contém 200 transações normais.

Dados já no formato correto para a API.

```bash
curl -X POST "http://localhost:8000/api/v1/fraud-detection/predict/batch" \
-H "Content-Type: application/json" \
-d '{
    "transactions": [
        {
            "Time": 0,
            "Amount": 100.0,
            "V1": 0.1,
            "V2": -0.1
            # ... demais características V3-V28
        },
        {
            "Time": 3600,
            "Amount": 50.0,
            "V1": 0.2,
            "V2": -0.2
            # ... demais características V3-V28
        }
    ]
}'
```


### Exemplo de Resposta

```json
{
    "count_fraud": 1,
    "count_non_fraud": 1,
    "results": [
        {
            "index": 0,
            "fraud_probability": 0.15,
            "prediction": 0
        },
        {
            "index": 1,
            "fraud_probability": 0.85,
            "prediction": 1
        }
    ]
}
```


## Treinamento do Modelo

O modelo foi treinado com os dados disponíveis no link abaixo.
As métricas do modelo foram salvas no arquivo `metrics/metrics.json`.

https://drive.google.com/file/d/1JeS9EO7vmF74yl1Qe-WS0xexMZaa3OYy/view?usp=sharing

Para executar o treinamento do modelo, o arquivo csv deve ser salvo no diretório data, respeitando a busca data/creditcard.csv

```bash
# Treinar o modelo
docker exec -it fraud-api poetry run python src/train.py
```


## Executar Testes

```bash
docker exec -it fraud-api poetry run pytest tests/ -v
```


### Tipos de Testes

- `test_api.py`: Testes da API
- `test_performance.py`: Testes de desempenho e tempo de resposta
- `test_train.py`: Testes do treinamento do modelo