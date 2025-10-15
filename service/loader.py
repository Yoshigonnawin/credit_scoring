import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import joblib
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor


logger = logging.getLogger("uvicorn.error")


class DataManager:
    """Управляет загрузкой и хранением всех данных"""

    def __init__(self):
        self.column_transformer: ColumnTransformer = None
        self.model: CatBoostRegressor = None

    def load_all(self, column_transformer_pth: Path, model_pth: Path) -> None:
        self._load_column_transformer(column_transformer_pth)
        self._load_model(model_pth)
        logger.info("All data loaded successfully")

    def _load_column_transformer(self, path: Path) -> None:
        logger.info(f"Loading column transformer from {path}")
        try:
            self.column_transformer = joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading column transformer from {path}: {e}")

    def _load_model(self, path: Path):
        logger.info(f"Loading column transformer from {path}")
        try:
            self.model = joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
