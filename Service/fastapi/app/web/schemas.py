from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: list[int]