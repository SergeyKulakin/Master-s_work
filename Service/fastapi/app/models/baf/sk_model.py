from pathlib import Path

import joblib
import numpy as np
import pandas as pd


model: object | None = None


def load_model(model_path: Path) -> None:
    global model
    model = joblib.load(model_path)


def unload_model() -> None:
    global model
    model = None


def run_model(df: pd.DataFrame, threshold: float = 0.8) -> list[int]:
    if model is None:
        raise RuntimeError("first load model")
    pred = model.predict(df.values)
    pred = np.where(pred > threshold, 1, 0)
    return pred.tolist()
