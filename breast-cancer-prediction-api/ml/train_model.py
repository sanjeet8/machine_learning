'''
Create the model training script
Now we'll train an ML model directly from our VS Code project. We'll use an SVM (Support Vector Machine)
inside a scikit-learn Pipeline, so feature scaling is learned correctly from the training data.
'''

from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "breast_cancer_pipeline.joblib"

def main():
    # 1. Load dataset
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target

    # 2. Split data before fitting preprocessing or model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 3. Build a pipeline: scaling happens before SVM prediction
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=1.0)),
        ]
    )

    # 4. Train the pipeline
    model.fit(X_train, y_train)

    # 5. Evaluate on the held-out test set
    predictions = model.predict(X_test)

    print("=== Model Evaluation ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(
        "Malignant recall: "
        f"{recall_score(y_test, predictions, pos_label=0):.4f}"
    )
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
        )
    )

    # 6. Save the complete pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
