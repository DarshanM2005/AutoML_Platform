# AutoML Platform - Module 3: Model Training and Evaluation
# File: ml_engine/training.py
#
# Author : Darshan M (AIML Intern - LearnDepth)
# Mentor : Ritesh Bonthalakoti

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ------------------------------------------------------------------
# Step 1: Make sure the data we received is valid before we do anything
# ------------------------------------------------------------------
def _validate_input(processed_data):

    if not isinstance(processed_data, dict):
        raise TypeError(
            f"processed_data should be a dictionary, but got {type(processed_data).__name__}."
        )

    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    missing = [key for key in required_keys if key not in processed_data]
    if missing:
        raise ValueError(f"The following keys are missing from the input: {missing}")

    if len(processed_data["X_train"]) == 0 or len(processed_data["y_train"]) == 0:
        raise ValueError(
            "Training data is empty. Please pass a dataset that has at least some rows."
        )


# ------------------------------------------------------------------
# Step 2: Define all the models we want to train
# ------------------------------------------------------------------
def _get_models():

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
    }

    return models


# ------------------------------------------------------------------
# Step 3: After training, check how well each model performed
# ------------------------------------------------------------------
def _evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    # Print the confusion matrix so we can visually see where the model went right/wrong
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix — {model_name}:")
    print(f"  {cm}")

    # Calculate the 4 standard metrics
    metrics = {
        "accuracy"        : round(accuracy_score(y_test, y_pred), 4),
        "precision"       : round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall"          : round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score"        : round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": cm.tolist()  # convert to list so it can be saved as JSON later
    }

    return metrics


# ------------------------------------------------------------------
# Main function — this is what the pipeline controller will call
# ------------------------------------------------------------------
def train_and_evaluate_models(processed_data):
    """
    Takes in the preprocessed data, trains 4 different ML models,
    evaluates each one, and returns the best performing model.

    Input:
        processed_data = {
            "X_train": ...,
            "X_test" : ...,
            "y_train": ...,
            "y_test" : ...
        }

    Output:
        {
            "best_model": <trained model>,
            "model_name": "Random Forest",
            "metrics"   : {
                "accuracy"        : 0.95,
                "precision"       : 0.94,
                "recall"          : 0.95,
                "f1_score"        : 0.94,
                "confusion_matrix": [...]
            }
        }
    """

    # First, make sure the data is in the right shape
    _validate_input(processed_data)

    X_train = processed_data["X_train"]
    X_test  = processed_data["X_test"]
    y_train = processed_data["y_train"]
    y_test  = processed_data["y_test"]

    # Get all models we want to try
    models = _get_models()

    results      = {}  # will store each model's result here
    failed       = []  # keep track of any models that crash during training

    print("\n" + "=" * 55)
    print("    AutoML — Training & Evaluation Started")
    print("=" * 55)

    # Train each model one by one
    for model_name, model in models.items():
        print(f"\n[Training]  {model_name} ...")

        try:
            # Train the model
            model.fit(X_train, y_train)

            # Evaluate it on the test set
            metrics = _evaluate_model(model, X_test, y_test, model_name)

            # Save the result
            results[model_name] = {
                "model"  : model,
                "metrics": metrics
            }

            # Show a quick summary
            print(f"\n[Done]      {model_name}")
            print(f"            Accuracy : {metrics['accuracy']}")
            print(f"            Precision: {metrics['precision']}")
            print(f"            Recall   : {metrics['recall']}")
            print(f"            F1 Score : {metrics['f1_score']}")

        except Exception as e:
            # If a model fails, don't stop — just log it and move on
            print(f"[Failed]    {model_name} could not be trained. Reason: {e}")
            failed.append(model_name)

    # If every single model failed, something is seriously wrong with the data
    if not results:
        raise RuntimeError(
            f"All models failed to train: {failed}. "
            "Please check your dataset for issues like wrong format or corrupted values."
        )

    # Pick the model with the highest F1 Score
    # (F1 is the most reliable metric when we don't know if the data is balanced or not)
    best_model_name = max(results, key=lambda name: results[name]["metrics"]["f1_score"])
    best            = results[best_model_name]

    print("\n" + "=" * 55)
    print(f"  Best Model : {best_model_name}")
    print(f"  F1 Score   : {best['metrics']['f1_score']}")
    print("=" * 55 + "\n")

    # Return in the exact format the Model Registry module expects
    return {
        "best_model": best["model"],
        "model_name": best_model_name,
        "metrics"   : best["metrics"]
    }


# ------------------------------------------------------------------
# Quick test — run this file directly to verify everything works
# python ml_engine/training.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    print("Running a quick test using the Iris dataset...\n")

    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    processed_data = {
        "X_train": X_train,
        "X_test" : X_test,
        "y_train": y_train,
        "y_test" : y_test
    }

    result = train_and_evaluate_models(processed_data)

    print("Final Output:")
    print(f"  Best Model : {result['model_name']}")
    print(f"  Metrics    : {result['metrics']}")