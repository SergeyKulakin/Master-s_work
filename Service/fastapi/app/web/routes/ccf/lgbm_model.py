import pandas as pd
from fastapi import APIRouter, File, UploadFile, Query
from fastapi import status

from app.models.ccf import lgbm_model as lgbm_model_module
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
    threshold: float = Query(0.8, gt=0., lt=1.),
) -> schemas.PredictionResponse:
    df = pd.read_csv(file.file)
    pred = lgbm_model_module.run_model(df, threshold=threshold)
    return schemas.PredictionResponse(
        prediction=pred,
    )
