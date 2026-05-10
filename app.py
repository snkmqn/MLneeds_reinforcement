import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prediction import NeedsReinforcementPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("needs_reinforcement_ml")

app = FastAPI(title="Needs Reinforcement ML Service")

try:
    predictor = NeedsReinforcementPredictor(
        model_path="model/needs_reinforcement.pkl"
    )
    logger.info(
        "Model loaded successfully: model=%s threshold=%.4f features_count=%d",
        predictor.model_name,
        float(predictor.threshold),
        len(predictor.all_features),
    )
except Exception:
    logger.exception("Failed to load needs reinforcement model")
    raise


class ReinforcementRequest(BaseModel):
    user_level: str
    learning_goal: str
    topic_code: str
    subtopic_code: Optional[str] = "topic_final"
    topic_level: str
    quiz_type: str

    quiz_score: float = Field(ge=0, le=100)
    avg_last_3_scores: Optional[float] = None
    previous_fails_same_topic: Optional[int] = 0
    subtopic_order: Optional[int] = 1

    preferred_topic_match: Optional[int] = 0
    completed_interactive: Optional[int] = 1


@app.get("/health")
def health():
    logger.info("Health check requested")

    return {
        "status": "ok",
        "model": predictor.model_name,
        "threshold": round(float(predictor.threshold), 4),
        "features": predictor.all_features,
    }


@app.post("/predict/reinforcement")
def predict_reinforcement(request: ReinforcementRequest):
    data = request.model_dump()

    logger.info(
        "Prediction request received: quiz_type=%s topic=%s subtopic=%s quiz_score=%.2f avg_last_3=%s previous_fails=%s",
        data.get("quiz_type"),
        data.get("topic_code"),
        data.get("subtopic_code"),
        data.get("quiz_score"),
        data.get("avg_last_3_scores"),
        data.get("previous_fails_same_topic"),
    )

    try:
        result = predictor.predict_single(data)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="PREDICTION_FAILED") from exc

    logger.info(
        "Prediction response: needs_reinforcement=%s prediction=%s probability=%.4f confidence=%.4f model=%s",
        result["needs_reinforcement"],
        result["prediction"],
        float(result["probability"]),
        float(result["confidence"]),
        result.get("model_name"),
    )

    return {
        "needs_reinforcement": result["needs_reinforcement"],
        "prediction": result["prediction"],
        "probability": result["probability"],
        "confidence": result["confidence"],
        "threshold": result.get("threshold"),
        "model_name": result.get("model_name"),
        "model_input": data,
    }