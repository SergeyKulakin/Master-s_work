from app.models.baf import sk_model as sk_module
from app.models.ccf import lgbm_model as lgbm_module
from app.models.ccf import custom_model as custom_module
from app.settings import settings


def load_models() -> None:
    sk_module.load_model(settings.stacking_classifier_file_path)
    lgbm_module.load_model(settings.lgbm_file_path)
    custom_module.load_model(settings.custom_file_path)

