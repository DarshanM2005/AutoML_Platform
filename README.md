# AutoML Platform — Module 3: Model Training and Evaluation

## Overview
This module is part of an Automated Machine Learning (AutoML) platform built during the LearnDepth AIML Internship.

Once a dataset has been cleaned and prepared by the preprocessing module, this module automatically trains multiple machine learning models, evaluates their performance, and returns the best-performing model for storage and prediction.

---

## Project Structure

```
AutoML_Platform/
├── ml_engine/
│   ├── __init__.py
│   └── training.py       ← Main module (Model Training & Evaluation)
├── requirements.txt
└── README.md
```

---

## Models Implemented

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| K-Nearest Neighbors | scikit-learn |
| XGBoost | xgboost |

---

## Evaluation Metrics

Each model is evaluated using the following metrics:

- **Accuracy** — Overall correctness of predictions
- **Precision** — Quality of positive predictions
- **Recall** — Coverage of actual positives
- **F1 Score** — Balanced measure of Precision and Recall
- **Confusion Matrix** — Visual breakdown of correct and incorrect predictions

The best model is selected based on the highest **F1 Score**.

---

## Input Format

The module expects preprocessed data in the following format:

```python
processed_data = {
    "X_train": X_train,
    "X_test" : X_test,
    "y_train": y_train,
    "y_test" : y_test
}
```

---

## Output Format

```python
{
    "best_model": <trained model object>,
    "model_name": "Random Forest",
    "metrics": {
        "accuracy"        : 0.95,
        "precision"       : 0.94,
        "recall"          : 0.95,
        "f1_score"        : 0.94,
        "confusion_matrix": [[...], [...]]
    }
}
```

---

## How to Run

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 2 — Run the module directly (quick test with Iris dataset):**
```bash
python ml_engine/training.py
```

**Step 3 — Integrate with pipeline:**
```python
from ml_engine.training import train_and_evaluate_models

result = train_and_evaluate_models(processed_data)
best_model = result["best_model"]
```

---

## Error Handling

The module handles the following errors gracefully:

- Empty training dataset
- Missing or invalid input keys
- Individual model training failures (skips and continues)
- Complete failure of all models

---

## Dependencies

```
scikit-learn==1.8.0
xgboost==3.2.0
pandas==3.0.1
numpy==2.4.3
```

Full list available in `requirements.txt`

---

## Author

**Darshan M**
AIML Intern — LearnDepth
Mentor: Ritesh Bonthalakoti
