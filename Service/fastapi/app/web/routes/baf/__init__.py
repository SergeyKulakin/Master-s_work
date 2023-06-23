from fastapi import APIRouter

from app.web.routes.baf.sk_model import router as sk_model_router
from app.web.routes.baf.custom_1_model import router as custom_1_model_router
from app.web.routes.baf.custom_2_model import router as custom_2_model_router
from app.web.routes.baf.custom_3_model import router as custom_3_model_router
from app.web.routes.baf.generate import router as generator_router

router = APIRouter(tags=["baf"])
router.include_router(sk_model_router, prefix="/sk")
router.include_router(custom_1_model_router, prefix="/custom1")
router.include_router(custom_2_model_router, prefix="/custom2")
router.include_router(custom_3_model_router, prefix="/custom3")
router.include_router(generator_router, prefix="/generate")
