import io

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.models.ccf import generate as generate_module

router = APIRouter()


@router.post(
    "/get_data",
    response_class=StreamingResponse,
)
async def generate_router(
        frac: float = Query(gt=0,),
) -> StreamingResponse:
    df = generate_module.run_model(frac)
    stream = io.StringIO()
    df.to_csv(stream)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",)
    response.headers["Content-Disposition"] = "attachment; filename=export.csv"
    return response
