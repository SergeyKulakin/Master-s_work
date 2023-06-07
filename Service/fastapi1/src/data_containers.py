from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request schema for predict endpoint."""

    path_file: str




class PredictResponse(BaseModel):
    """Response schema for predict endpoint."""

    pred_class: int
    error: str = ''


