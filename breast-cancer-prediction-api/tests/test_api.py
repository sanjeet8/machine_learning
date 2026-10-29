from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() =={"message": "ML Prediction API is running"}

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }

def test_predict_with_valid_features():
    # One known sample from the dataset
    features = [
        17.99, 10.38, 122.8, 1001.0, 0.1184,
        0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
    ]

    response = client.post(
        "/predict",
        json={"features": features},
        )

    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] == 0
    assert body["predicted_label"] == "malignant"

def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [17.99, 10.38]},)

    assert response.status_code == 422
