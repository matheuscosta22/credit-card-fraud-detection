from fastapi import APIRouter, FastAPI
from src.types.transaction import Transaction, TransactionsBatch
from src.services import fraudDetection

app = FastAPI(title="Fraud Detection API")
router = APIRouter(prefix="/api/v1/fraud-detection")

@router.get("/")
def root()-> dict:
    return {"status": "API is running"}

@router.post("/predict")
def predict(transaction: Transaction)-> dict:
    return fraudDetection.predict(transaction)

@router.post("/predict/batch")
def predict_batch(batch: TransactionsBatch)-> dict:
    return fraudDetection.predict_batch(batch.transactions)


app.include_router(router)