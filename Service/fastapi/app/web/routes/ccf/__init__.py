from fastapi import APIRouter

from app.web.routes.ccf.lgbm_model import router as lgbm_router
from app.web.routes.ccf.custom_model import router as custom_router
from app.web.routes.ccf.plots import router as plots_router
from app.web.routes.ccf.generate import router as generate_router

router = APIRouter(tags=["ccf"])
router.include_router(lgbm_router, prefix="/lgbm")
router.include_router(custom_router, prefix="/custom")
router.include_router(plots_router, prefix="/plots")
router.include_router(generate_router, prefix="/generate")
