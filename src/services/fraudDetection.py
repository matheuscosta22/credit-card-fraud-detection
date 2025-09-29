from typing import List, Generator, Dict
import pandas as pd
from src.train import addFeatures
from src.types.transaction import Transaction
import asyncio
from src.services.requestManager import RequestManager
from src.services.loadModel import load_models

CHUNK_SIZE = 50
SUBCHUNK_SIZE = CHUNK_SIZE // 4


modelXgb, modelScaler = load_models()
requestManager = RequestManager()


def predict(transaction: Transaction) -> dict:
    transactionData = addFeatures(
        pd.DataFrame([transaction.model_dump()])
    )

    featuresScaled = modelScaler.transform(transactionData)

    fraudProbability = modelXgb.predict_proba(featuresScaled)[0][1]
    fraudPrediction = modelXgb.predict(featuresScaled)[0]

    return {
        "fraud_probability": float(fraudProbability),
        "prediction": int(fraudPrediction)
    }


def process_chunk(chunkData: pd.DataFrame) -> List[Dict]:
    processedData = addFeatures(chunkData)
    featuresScaled = modelScaler.transform(processedData)
    
    fraudProbabilities = modelXgb.predict_proba(featuresScaled)[:, 1]
    fraudPredictions = modelXgb.predict(featuresScaled)
    
    return [
        {
            "index": int(index),
            "fraud_probability": float(probability),
            "prediction": int(prediction)
        }
        for index, (probability, prediction) in zip(processedData.index, zip(fraudProbabilities, fraudPredictions))
    ]


async def process_subchunk(subChunkData: pd.DataFrame, chunkStartIndex: int) -> List[Dict]:
    subChunks = []
    for subChunkStart in range(0, len(subChunkData), SUBCHUNK_SIZE):
        currentChunk = subChunkData[subChunkStart:subChunkStart + SUBCHUNK_SIZE]

        currentChunk.index = range(
            chunkStartIndex + subChunkStart,
            chunkStartIndex + subChunkStart + len(currentChunk)
        )
        subChunks.append(currentChunk)

    return subChunks


async def predict_batch(transactions: List[Transaction]) -> Generator[dict, None, None]:
    try:
        totalFraud = 0
        totalNonFraud = 0
        processedCount = 0
        transactionsCount = len(transactions)

        for chunkStartIndex in range(0, transactionsCount, CHUNK_SIZE):
            currentChunk = transactions[chunkStartIndex:chunkStartIndex + CHUNK_SIZE]

            chunkData = pd.DataFrame([t.model_dump() for t in currentChunk])
            
            subChunks = await process_subchunk(chunkData, chunkStartIndex)

            chunkResults = []
            for subChunk in subChunks:

                while requestManager.too_many_processes():
                    await asyncio.sleep(0.1)
                
                futureResult = asyncio.get_event_loop().run_in_executor(
                    requestManager.threadPool,
                    process_chunk,
                    subChunk
                )
                chunkResults.append(futureResult)


            completedResults = await asyncio.gather(*chunkResults)
            processedResults = [item for sublist in completedResults for item in sublist]

            batchPredictions = [result["prediction"] for result in processedResults]
            fraudCount = sum(batchPredictions)
            nonFraudCount = len(batchPredictions) - fraudCount

            totalFraud += fraudCount
            totalNonFraud += nonFraudCount
            processedCount += len(batchPredictions)
            
            yield {
                "partial_results": {
                    "processed_count": processedCount,
                    "total_count": transactionsCount,
                    "fraud_count": int(totalFraud),
                    "non_fraud_count": int(totalNonFraud),
                    "results_count": len(processedResults),
                    "results": processedResults
                }
            }
    finally:
        await requestManager.release_slot()