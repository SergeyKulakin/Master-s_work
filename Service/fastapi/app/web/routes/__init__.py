from fastapi import APIRouter

from app.web.routes.ccf import router as ccf_router
from app.web.routes.baf import router as baf_router

router = APIRouter()
router.include_router(ccf_router, prefix="/ccf")
router.include_router(baf_router, prefix="/baf")

