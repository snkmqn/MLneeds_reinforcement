from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from prediction import NeedsReinforcementPredictor


app = FastAPI(title="Needs Reinforcement ML Service")

predictor = NeedsReinforcementPredictor(
    model_path="model/needs_reinforcement.pkl"
)


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
    return {
        "status": "ok",
        "model": predictor.model_name,
        "threshold": round(float(predictor.threshold), 4),
        "features": predictor.all_features,
    }


@app.post("/predict/reinforcement")
def predict_reinforcement(request: ReinforcementRequest):
    data = request.model_dump()

    if data.get("subtopic_code") is None or data.get("subtopic_code") == "":
        data["subtopic_code"] = "topic_final"

    if data.get("avg_last_3_scores") is None:
        data["avg_last_3_scores"] = data["quiz_score"]

    if data.get("previous_fails_same_topic") is None:
        data["previous_fails_same_topic"] = 0

    if data.get("subtopic_order") is None:
        data["subtopic_order"] = 1

    if data.get("preferred_topic_match") is None:
        data["preferred_topic_match"] = 0

    if data.get("completed_interactive") is None:
        data["completed_interactive"] = 1

    result = predictor.predict_single(data)

    return {
        "needs_reinforcement": result["needs_reinforcement"],
        "prediction": result["prediction"],
        "probability": result["probability"],
        "confidence": result["confidence"],
        "threshold": result.get("threshold"),
        "model_name": result.get("model_name"),
        "model_input": data,
    }