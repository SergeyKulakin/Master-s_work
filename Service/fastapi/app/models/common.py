from app.models.baf import sk_model as sk_module
from app.models.baf import custom_1_model as custom_1_module
#from app.models.baf import custom_2_model as custom_2_module
from app.models.baf import custom_3_model as custom_3_module
from app.models.baf import generate as generate_module_baf
from app.models.ccf import lgbm_model as lgbm_module
from app.models.ccf import custom_model as custom_module
from app.models.ccf import generate as generate_module_ccf
from app.settings import settings


def load_models() -> None:
    sk_module.load_model(settings.stacking_classifier_file_path)
    lgbm_module.load_model(settings.lgbm_file_path)
    custom_module.load_model(settings.custom_file_path)
    custom_1_module.load_model(settings.custom_1_baf_path)
    #custom_2_module.load_model(settings.custom_2_baf_path)
    custom_3_module.load_model(settings.custom_3_baf_path)
    generate_module_baf.load_model(settings.generate_model_baf_path, settings.generate_qt_baf_path)
    generate_module_ccf.load_model(settings.generate_model_ccf_path, settings.generate_qt_ccf_path)

