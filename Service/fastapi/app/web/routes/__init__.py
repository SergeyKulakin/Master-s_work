from fastapi import APIRouter

from app.web.routes.ccf import router as ccf_router

router = APIRouter()
router.include_router(ccf_router, prefix="/ccf")

