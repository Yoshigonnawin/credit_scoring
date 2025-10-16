import logging
from pathlib import Path
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request
import pandas as pd

from .loader import DataManager
from .preprocessor import Preprocessor
from .predictor import Predictor


# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("uvicorn.error")


COLUMN_TRANSFORMER_PATH = os.getenv(
    "COLUMN_TRANSFORMER_PATH", Path("model/column_transformer.pkl")
)
MODEL_PATH = os.getenv("MODEL_PATH", Path("model/model.pkl"))
TEST_DATA_PATH = os.getenv(
    "TEST_DATA_PATH", Path("test_data/test_data.parquet")
)
CUTOFF = os.getenv("CUTOFF", 363)
CUTOFF = int(CUTOFF)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Starting application...")
    try:
        # Сохраняем экземпляр в app.state для использования в endpoint'ах
        app.state.data_manager = DataManager()
        app.state.data_manager.load_all(
            COLUMN_TRANSFORMER_PATH, MODEL_PATH, TEST_DATA_PATH
        )
        app.state.preprocessor = Preprocessor(app.state.data_manager.column_transformer)
        app.state.predictor = Predictor(cutoff=CUTOFF)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="ML Prediction Service",
    description="Микросервис для предсказаний с использованием CatBoost",
    version="0.1",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy"}


@app.get("/get_test_resolution")
async def get_test_resolution():
    try:
        res = app.state.predictor.get_test_predictions(
            model=app.state.data_manager.model,
            preprocessor=app.state.preprocessor,
            data=app.state.data_manager.test_data,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get test resolution: {e}"
        )
    return res


@app.post("/get_resolution")
async def get_resolution(request: Request):
    """Получение предсказания модели"""
    try:
        body = await request.json()
        input_data = body.get("input_data")
        if input_data is None:
            raise ValueError("Missing 'input_data' key in JSON")

        dat = pd.DataFrame([input_data])

        res = app.state.predictor.get_predictions(
            model=app.state.data_manager.model,
            preprocessor=app.state.preprocessor,
            data=dat,
        )
        return res

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resolution: {e}")
