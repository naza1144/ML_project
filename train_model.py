"""Train a classifier that predicts which coffee menu item will be ordered
given only the time of the order (weekday, month, time of day) -- no
price/leakage features are used since price is a consequence of the order,
not a cause.

Feature/model choice notes (see exploration in this session):
- hour_of_day / Time_of_Day are redundant; the coarser Time_of_Day bucket
  generalizes better than raw hour (less overfitting on a 24-way split).
- GradientBoostingClassifier beat RandomForest/LogisticRegression on this
  feature set and beat class-balanced weighting on both accuracy and macro-F1.
- The time-only signal is weak: the majority-class baseline (always guess
  "Americano with Milk") already scores ~0.228 accuracy. This model reaches
  ~0.29, a real but modest lift -- coffee choice is only loosely tied to
  weekday/month/time-of-day in this dataset.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TOP_K = 4

FEATURES = ["Weekday", "Month_name", "Time_of_Day"]
TARGET = "coffee_name"

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TIME_OF_DAY_ORDER = ["Morning", "Afternoon", "Night"]


def load_data(path="Coffe_sales.csv"):
    df = pd.read_csv(path)
    return df[FEATURES + [TARGET]]


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )
    clf = GradientBoostingClassifier(random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    baseline_acc = y.value_counts(normalize=True).max()
    top_k_acc = top_k_accuracy_score(
        y_test, y_proba, k=TOP_K, labels=pipeline.named_steps["model"].classes_
    )
    print(f"Majority-class baseline accuracy: {baseline_acc:.3f}")
    print(f"Model test accuracy (Top-1):       {acc:.3f}")
    print(f"Model test accuracy (Top-{TOP_K}):       {top_k_acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    joblib.dump(
        {
            "pipeline": pipeline,
            "classes": sorted(y.unique()),
            "weekday_order": WEEKDAY_ORDER,
            "month_order": MONTH_ORDER,
            "time_of_day_order": TIME_OF_DAY_ORDER,
            "test_accuracy": acc,
            "baseline_accuracy": baseline_acc,
            "top_k": TOP_K,
            "top_k_accuracy": top_k_acc,
        },
        "model/coffee_model.joblib",
    )
    print("\nSaved model to model/coffee_model.joblib")


if __name__ == "__main__":
    main()
