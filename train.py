import joblib
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/needs_reinforcement_synthetic_20000_v2.csv")
MODEL_PATH = Path("model/needs_reinforcement.pkl")

TARGET = "needs_reinforcement"

CATEGORICAL_FEATURES = [
    "user_level",
    "learning_goal",
    "topic_code",
    "subtopic_code",
    "topic_level",
    "quiz_type",
]

NUMERICAL_FEATURES = [
    "quiz_score",
    "avg_last_3_scores",
    "previous_fails_same_topic",
    "subtopic_order",
]

BINARY_FEATURES = [
    "preferred_topic_match",
    "completed_interactive",
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "binary",
                "passthrough",
                BINARY_FEATURES,
            ),
        ]
    )


def predict_with_threshold(model, X, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions


def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    probabilities, predictions = predict_with_threshold(model, X_test, threshold)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("=" * 80)
    print(f"MODEL: {name}")
    print(f"THRESHOLD: {threshold:.2f}")
    print("=" * 80)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def find_best_threshold(model, X_test, y_test):

    best_threshold = 0.50
    best_f1 = -1.0
    best_recall = -1.0

    print("=" * 80)
    print("THRESHOLD SEARCH")
    print("=" * 80)

    for threshold in [x / 100 for x in range(30, 56, 5)]:
        _, predictions = predict_with_threshold(model, X_test, threshold)

        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        print(
            f"threshold={threshold:.2f} "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"f1={f1:.4f}"
        )


        if recall >= 0.80:
            if f1 > best_f1 or (abs(f1 - best_f1) < 1e-9 and recall > best_recall):
                best_f1 = f1
                best_recall = recall
                best_threshold = threshold

    if best_f1 < 0:
        for threshold in [x / 100 for x in range(30, 56, 5)]:
            _, predictions = predict_with_threshold(model, X_test, threshold)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_recall = recall
                best_threshold = threshold

    print()
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Recall at best threshold: {best_recall:.4f}")
    print()

    return best_threshold


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(df.head())
    print()
    print("Target distribution:")
    print(df[TARGET].value_counts())
    print()
    print("Target distribution normalized:")
    print(df[TARGET].value_counts(normalize=True))
    print()

    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES

    X = df[feature_columns]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=14,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    trained_models = {}

    for name, estimator in models.items():
        preprocessor = build_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        print("=" * 80)
        print(f"TRAINING: {name}")
        print("=" * 80)

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            name=name,
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            threshold=0.50,
        )

        trained_models[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
        }

    candidate_names = [
        name for name, item in trained_models.items()
        if item["metrics"]["recall"] >= 0.80
    ]

    if candidate_names:
        selected_model_name = max(
            candidate_names,
            key=lambda name: trained_models[name]["metrics"]["f1"],
        )
    else:
        selected_model_name = max(
            trained_models.keys(),
            key=lambda name: trained_models[name]["metrics"]["f1"],
        )

    selected_pipeline = trained_models[selected_model_name]["pipeline"]

    best_threshold = find_best_threshold(
        model=selected_pipeline,
        X_test=X_test,
        y_test=y_test,
    )

    final_metrics = evaluate_model(
        name=selected_model_name,
        model=selected_pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=best_threshold,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "model": selected_pipeline.named_steps["model"],
        "preprocessor": selected_pipeline.named_steps["preprocessor"],
        "threshold": best_threshold,
        "model_name": selected_model_name,
        "metrics": final_metrics,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "target": TARGET,
    }

    joblib.dump(package, MODEL_PATH)

    print("=" * 80)
    print("MODEL SAVED")
    print("=" * 80)
    print(f"Path: {MODEL_PATH}")
    print(f"Model: {selected_model_name}")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Metrics: {final_metrics}")


if __name__ == "__main__":
    main()
