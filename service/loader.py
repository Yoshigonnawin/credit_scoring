import logging
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
import pandas as pd


logger = logging.getLogger("uvicorn.error")


class DataManager:
    """Управляет загрузкой и хранением всех данных"""

    def __init__(self):
        self.column_transformer: ColumnTransformer = None
        self.model: CatBoostRegressor = None
        self.test_data: pd.DataFrame = None

    def load_all(
        self, column_transformer_pth: Path, model_pth: Path, test_data_pth: Path
    ) -> None:
        self._load_column_transformer(column_transformer_pth)
        self._load_model(model_pth)
        self._load_test_data(test_data_pth)
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

    def _load_test_data(self, path: Path):
        logger.info(f"Loading test data from:{path}")
        try:
            self.test_data = pd.read_parquet(path)
        except Exception as e:
            logger.error(f"Error loading test data from:{path}: {e}")
