from pathlib import Path

from pydantic import BaseSettings, Extra


class Settings(BaseSettings):
    lgbm_file_path: Path = "models/LGBM.pickle"
    custom_file_path: Path = "models/Custom.pth"
    stacking_classifier_file_path: Path = "models/StackingClassifier.pickle"

    class Config(BaseSettings.Config):
        env_file = ".env"
        extra = Extra.ignore


settings = Settings()
