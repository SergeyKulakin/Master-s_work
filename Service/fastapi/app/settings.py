from pathlib import Path

from pydantic import BaseSettings, Extra


class Settings(BaseSettings):
    lgbm_file_path: Path = "models/LGBM.pickle"
    custom_file_path: Path = "models/Custom.pth"
    stacking_classifier_file_path: Path = "models/StackingClassifier.pickle"
    custom_1_baf_path: Path = "models/Custom_1_BAF.pth"
    #custom_2_baf_path: Path = ""  # FIXME: set path
    custom_3_baf_path: Path = "models/Custom_3_BAF.pth"

    generate_model_baf_path: Path = "models/GAN_BAF.pth"
    generate_qt_baf_path: Path = "models/QT_BAF.pkl"
    generate_model_ccf_path: Path = "models/CCF_GAN.pth"
    generate_qt_ccf_path: Path = "models/QT_CCF.pkl"

    class Config(BaseSettings.Config):
        env_file = ".env"
        extra = Extra.ignore


settings = Settings()
