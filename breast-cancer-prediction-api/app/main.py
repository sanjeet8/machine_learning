'''
Create the FastAPI application
So far, your project can train and save a model. Now we'll make it accessible through HTTP,
just like a backend service you'd build at work.

The flow will be:
Client (Sends 30 feature values)
               |
               V
FastAPI (Validates the HTTP request)
               |
               V
Saved ML pipeline (Scales features and predicts)
               |
               V
JSON response (Returns the predicted class)
'''

from sklearn.datasets import load_breast_cancer
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Locate the saved model relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_pipeline.joblib"

app = FastAPI(
    title="ML Prediction API",
    description="An API that serves a trained machine-learning model.",
    version="1.0.0",
)

# Load the trained pipeline once when the application starts
if not MODEL_PATH.exists:
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Run 'python ml/train_model.py' first."
    )

model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Feature names mapped to numerical values.",
    )

    @classmethod
    def feature_names(cls) -> list[str]:
        dataset = load_breast_cancer()
        return list(dataset.feature_names)

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str

@app.get("/")
def home():
    return {"message": "ML Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = int(model.predict([request.features])[0])
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid feature values.",
        ) from exc

    # In this dataset: 0 = malignant, 1 = benign
    label = "malignant" if prediction == 0 else "benign"

    return PredictionResponse(
        predicted_class=prediction,
        predicted_label=label,
    )

'''
for testing in /predict endpoint
{
  "features": [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}
'''

'''
5.2 Understand the important parts
- FastAPI(...) creates the web application.
- joblib.load(...) loads the model once rather than retraining for every request.
- PredictionRequest uses Pydantic to validate incoming JSON.
- @app.get("/health") provides a health-check endpoint.
- @app.post("/predict") accepts input and returns the prediction.
- response_model=PredictionResponse defines and validates the response structure.
Important: The API requires exactly 30 features in the same order used during training.
We'll make this safer and easier to use in a later step.

To run: 
From your project root, run:
python -m uvicorn app.main:app --reload
'''