import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_random_forest_model.joblib"
FEATURE_PATH = MODEL_DIR / "feature_columns.json"
LABEL_PATH = MODEL_DIR / "label_classes.json"

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

app = FastAPI(title="Student Dropout Predictor")

model = joblib.load(MODEL_PATH)
feature_columns = json.loads(FEATURE_PATH.read_text(encoding="utf-8"))
label_classes = json.loads(LABEL_PATH.read_text(encoding="utf-8"))


def preprocess_record(record: dict) -> pd.DataFrame:
    data = record.copy()
    data.pop("Target", None)

    for col in CATEGORICAL_FEATURES:
        if col not in data:
            data[col] = None

    df = pd.DataFrame([data])

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    numerical_features = [col for col in df.columns if col not in CATEGORICAL_FEATURES]
    X_categorical_processed = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)
    X_processed = pd.concat([df[numerical_features], X_categorical_processed], axis=1)

    X_processed = X_processed.reindex(columns=feature_columns, fill_value=0)
    return X_processed


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict) -> dict:
    X = preprocess_record(payload)
    probs = model.predict_proba(X)[0]
    pred_index = int(probs.argmax())

    return {
        "predicted_class_index": pred_index,
        "predicted_label": label_classes[pred_index],
        "probabilities": dict(zip(label_classes, probs.tolist())),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
