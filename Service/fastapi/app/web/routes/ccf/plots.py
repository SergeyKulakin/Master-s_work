import io

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import StreamingResponse

from app.models.ccf.plots import make_hists, make_count_frauds

router = APIRouter()


@router.post(
    "/hists",
    response_class=StreamingResponse,
)
async def plot_hists(
    file: UploadFile = File(media_type="text/csv"),
    lines_to_proceed: int = Query(0, description="Lines to proceed in file(except for column names). 0 to proceed all", ge=0),
) -> StreamingResponse:
    df = pd.read_csv(file.file)
    if lines_to_proceed == 0:
        lines_to_proceed = len(df)
    fig = make_hists(df, limit=lines_to_proceed)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@router.post(
    "/count_fraud",
    response_class=StreamingResponse,
)
async def count_fraud(
        file: UploadFile = File(media_type="text/csv"),
) -> StreamingResponse:
    df = pd.read_csv(file.file)
    fig = make_count_frauds(df)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

