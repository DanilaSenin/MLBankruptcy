"""ML-сервис (FastAPI) поверх Predictor. Бэкенд на TypeScript ходит сюда по HTTP.

  pip install fastapi uvicorn python-multipart
  set ML_ARTIFACTS=artifacts
  set ML_DATA_DIR=C:/haha/Tcache
  uvicorn ml_service:app --host 127.0.0.1 --port 8001

Документация и ручная проверка: http://127.0.0.1:8001/docs
"""
from __future__ import annotations

import io
import os
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from predictor import Predictor

MAX_CSV_BYTES = 5_000_000

P = Predictor(os.environ.get("ML_ARTIFACTS", "artifacts"), os.environ.get("ML_DATA_DIR") or None)
app = FastAPI(title="Bankruptcy ML service", version=P.cfg["version"])


class FeaturesIn(BaseModel):
    id: int | None = None
    features: dict[str, float | None] = Field(..., description="feature_0 … feature_1366")
    word_count: str | None = Field(None, description="текст в формате исходного файла: 'слово число слово число'")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", **P.info}


@app.get("/predict/{index}")
def predict_index(index: int, top_k: int = Query(5, ge=1, le=50), n_words: int = Query(10, ge=0, le=50)):
    try:
        return P.by_index(index, top_k, n_words)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/predict")
def predict_features(body: FeaturesIn, top_k: int = Query(5, ge=1, le=50), n_words: int = Query(10, ge=0, le=50)):
    if not body.features:
        raise HTTPException(422, "features пустой")
    return P.by_features(body.features, body.word_count, body.id, top_k, n_words)


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=50),
                      n_words: int = Query(10, ge=0, le=50)):
    raw = await file.read()
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(413, "файл больше 5 МБ")
    try:
        res = P.by_csv(io.BytesIO(raw), top_k, n_words)
    except Exception as e:  # битый CSV
        raise HTTPException(422, f"не удалось разобрать CSV: {e}") from e
    if not res:
        raise HTTPException(422, "в CSV нет строк")
    return res[0] if len(res) == 1 else res
