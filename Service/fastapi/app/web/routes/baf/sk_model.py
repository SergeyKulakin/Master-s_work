import pandas as pd
from fastapi import APIRouter, File, UploadFile, Query
from fastapi import status

from app.models.baf import sk_model as sk_model_module
from app.web import schemas

router = APIRouter()


@router.post(
    "/predict",
    responses={
        status.HTTP_200_OK: {
            "model": schemas.PredictionResponse,
            "description": "result of prediction",
        },
    },
)
async def predict_router(
        file: UploadFile = File(
            media_type="text/csv",
        ),
) -> schemas.PredictionResponse:
    df = pd.read_csv(file.file)
    pred = sk_model_module.run_model(df)
    return schemas.PredictionResponse(
        prediction=pred,
    )
