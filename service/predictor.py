from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import logging
from .preprocessor import Preprocessor
from random import choice


logger = logging.getLogger("uvicorn.error")


class Predictor:
    def __init__(self, cutoff: int, test_data: pd.DataFrame) -> None:
        self.cutoff = cutoff
        self.test_data = test_data

    def _predict_by_model(
        self, model: CatBoostRegressor, data: pd.DataFrame
    ) -> np.ndarray:
        return model.predict_proba(data)

    def _proba_2_score(self, proba: np.ndarray) -> np.ndarray:
        scores = (proba[:, 1] * 1000).astype(np.int16)
        return scores

    def _get_resolution_by_cutoff(
        self, scores: np.ndarray
    ) -> list[dict[str, int | str]]:
        return [
            {
                "score": int(score),
                "resolution": ("Approved" if score >= self.cutoff else "Decline"),
            }
            for score in scores
        ]

    def get_predictions(
        self, model: CatBoostRegressor, preprocessor: Preprocessor, data: pd.DataFrame
    ) -> list[dict[str, int | str]]:
        result = []
        try:
            data_prep = preprocessor.transform_data(data)
            proba = self._predict_by_model(model=model, data=data_prep)
            scores = self._proba_2_score(proba)
            result = self._get_resolution_by_cutoff(scores)
        except Exception as e:
            logger.error(f"Error in Predictor.get_predictions(), data:{data}: {e}")
        return result

    def get_test_predictions(
        self, model: CatBoostRegressor, preprocessor: Preprocessor
    ) -> list[dict[str, int | str]]:
        sel = self.test_data.iloc[choice(range(self.test_data.shape[0]))]
        return self.get_predictions(model, preprocessor, sel)
