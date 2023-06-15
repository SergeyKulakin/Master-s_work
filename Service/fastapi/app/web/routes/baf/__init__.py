from fastapi import APIRouter

from app.web.routes.baf.sk_model import router as sk_model_router

router = APIRouter(tags=["baf"])
router.include_router(sk_model_router, prefix="/sk")
