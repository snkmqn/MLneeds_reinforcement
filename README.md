# Needs Reinforcement ML Service

This project is an ML service for predicting whether a user needs additional reinforcement after completing a financial literacy quiz.

The model analyzes quiz results and user learning context, then returns whether the user should repeat the material or continue to the next lesson.

## Project Structure

```text
MLneeds_reinforcement/
├── app.py
├── train.py
├── prediction.py
├── requirements.txt
├── data/
│   └── needs_reinforcement_synthetic_20000_v2.csv
├── model/
│   └── needs_reinforcement.pkl
└── README.md
```

## Requirements

- Python 3.10+
- pip
- Git

Check Python version:

```bash
python --version
```

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/MLneeds_reinforcement.git
cd MLneeds_reinforcement
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model again, run:

```bash
python train.py
```

The trained model will be saved to:

```text
model/needs_reinforcement.pkl
```

## Running the API

Start the service:

```bash
python app.py
```

The API will be available locally, for example:

```text
http://127.0.0.1:5000
```

## Prediction Endpoint

Example endpoint:

```text
POST http://127.0.0.1:5000/predict
```

Example request body:

```json
{
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
  "completed_interactive": 0
}
```

Example response:

```json
{
  "prediction": 1,
  "needs_reinforcement": true,
  "probability": 0.7214,
  "confidence": 0.7214,
  "threshold": 0.45,
  "model_name": "Random Forest"
}
```

## Notes

The project uses a synthetic dataset for training and testing:

```text
data/needs_reinforcement_synthetic_20000_v2.csv
```

Real user personal data is not stored in this repository.
