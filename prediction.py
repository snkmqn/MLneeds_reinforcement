import joblib
import pandas as pd


class NeedsReinforcementPredictor:
    def __init__(self, model_path: str = "model/needs_reinforcement.pkl"):
        package = joblib.load(model_path)

        self.model = package["model"]
        self.preprocessor = package["preprocessor"]

        self.threshold = package.get("threshold", 0.5)

        self.categorical_features = package["categorical_features"]
        self.numerical_features = package["numerical_features"]
        self.binary_features = package["binary_features"]

        self.all_features = (
            self.categorical_features
            + self.numerical_features
            + self.binary_features
        )

        self.model_name = package.get("model_name", "Unknown")
        self.metrics = package.get("metrics", {})

        print("=" * 60)
        print("MODEL LOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"Model Type : {self.model_name}")
        print(f"Threshold  : {self.threshold:.4f}")
        print(f"Features   : {self.all_features}")
        print("=" * 60)

    def _prepare_dataframe(self, user_data: dict) -> pd.DataFrame:
        data = dict(user_data)

        if data.get("subtopic_code") is None or data.get("subtopic_code") == "":
            data["subtopic_code"] = "topic_final"

        if "quiz_type" not in data or data["quiz_type"] is None:
            data["quiz_type"] = "subtopic_quiz"

        if "avg_last_3_scores" not in data or data["avg_last_3_scores"] is None:
            data["avg_last_3_scores"] = data.get("quiz_score", 0)

        if "previous_fails_same_topic" not in data or data["previous_fails_same_topic"] is None:
            data["previous_fails_same_topic"] = 0

        if "subtopic_order" not in data or data["subtopic_order"] is None:
            data["subtopic_order"] = 1

        if "preferred_topic_match" not in data or data["preferred_topic_match"] is None:
            data["preferred_topic_match"] = 0

        if "completed_interactive" not in data or data["completed_interactive"] is None:
            data["completed_interactive"] = 1

        missing_features = [
            feature for feature in self.all_features
            if feature not in data
        ]

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        df = pd.DataFrame([data])
        return df[self.all_features]

    def predict_single(self, user_data: dict):
        df = self._prepare_dataframe(user_data)

        X_processed = self.preprocessor.transform(df)

        probability = self.model.predict_proba(X_processed)[0][1]
        prediction = int(probability >= self.threshold)
        confidence = probability if prediction == 1 else 1 - probability
        return {
            "prediction": prediction,
            "needs_reinforcement": bool(prediction),
            "probability": round(float(probability), 4),
            "confidence": round(float(confidence), 4),
            "threshold": round(float(self.threshold), 4),
            "model_name": self.model_name,
        }

    def predict_batch(self, users_list: list):
        results = []

        for user_data in users_list:
            results.append(self.predict_single(user_data))

        return results


if __name__ == "__main__":
    predictor = NeedsReinforcementPredictor()

    weak_topic_final_user = {
        "user_level": "beginner",
        "learning_goal": "saving_money",
        "topic_code": "budgeting",
        "subtopic_code": "topic_final",
        "topic_level": "beginner",
        "quiz_type": "topic_final_quiz",

        "quiz_score": 50,
        "avg_last_3_scores": 50,
        "previous_fails_same_topic": 0,
        "subtopic_order": 5,

        "preferred_topic_match": 0,
        "completed_interactive": 0,
    }

    strong_subtopic_user = {
        "user_level": "intermediate",
        "learning_goal": "general_improvement",
        "topic_code": "budgeting",
        "subtopic_code": "income_expenses",
        "topic_level": "beginner",
        "quiz_type": "subtopic_quiz",

        "quiz_score": 90,
        "avg_last_3_scores": 85,
        "previous_fails_same_topic": 0,
        "subtopic_order": 1,

        "preferred_topic_match": 1,
        "completed_interactive": 1,
    }

    print("\nWEAK TOPIC FINAL USER:")
    print(predictor.predict_single(weak_topic_final_user))

    print("\nSTRONG SUBTOPIC USER:")
    print(predictor.predict_single(strong_subtopic_user))