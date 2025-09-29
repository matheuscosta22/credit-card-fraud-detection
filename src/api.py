from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from src.types.transaction import Transaction, TransactionsBatch
from src.services import fraudDetection
from src.services.requestManager import RequestManager
import json

app = FastAPI(title="Fraud Detection API")
router = APIRouter(prefix="/api/v1/fraud-detection")

@router.get("/")
def root()-> dict:
    return {"status": "API is running"}

@router.post("/predict")
def predict(transaction: Transaction)-> dict:
    return fraudDetection.predict(transaction)

@router.post("/predict/batch")
async def predict_batch(batch: TransactionsBatch):

    try:
        if not await RequestManager().acquire_slot():
            return StreamingResponse(
                iter([json.dumps({"error": "Server is processing too many requests, try again later"}) + "\n"]),
                status_code=500,
                media_type="application/x-ndjson"
            )


        async def generate_response():
            try:
                async for result in fraudDetection.predict_batch(batch.transactions):
                    yield json.dumps(result) + "\n"
            except Exception as error:
                yield json.dumps({"error": str(error)}) + "\n"


        return StreamingResponse(
            generate_response(),
            media_type="application/x-ndjson"
        )

    except Exception as error:
        print(error)
        return StreamingResponse(
            iter([json.dumps({"error": "Internal server error"}) + "\n"]),
            status_code=500,
            media_type="application/x-ndjson"
        )


app.include_router(router)