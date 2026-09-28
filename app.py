import os

from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "coffee_model.joblib")
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
classes = bundle["classes"]
weekday_order = bundle["weekday_order"]
month_order = bundle["month_order"]
time_of_day_order = bundle["time_of_day_order"]
test_accuracy = bundle["test_accuracy"]
baseline_accuracy = bundle["baseline_accuracy"]
top_k = bundle["top_k"]
top_k_accuracy = bundle["top_k_accuracy"]


@app.route("/")
def index():
    return render_template(
        "index.html",
        weekdays=weekday_order,
        months=month_order,
        times_of_day=time_of_day_order,
        test_accuracy=round(test_accuracy * 100, 1),
        baseline_accuracy=round(baseline_accuracy * 100, 1),
        top_k=top_k,
        top_k_accuracy=round(top_k_accuracy * 100, 1),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    weekday = data.get("weekday")
    month = data.get("month")
    time_of_day = data.get("time_of_day")

    if weekday not in weekday_order:
        return jsonify({"error": f"invalid weekday: {weekday}"}), 400
    if month not in month_order:
        return jsonify({"error": f"invalid month: {month}"}), 400
    if time_of_day not in time_of_day_order:
        return jsonify({"error": f"invalid time_of_day: {time_of_day}"}), 400

    row = pd.DataFrame(
        [{"Weekday": weekday, "Month_name": month, "Time_of_Day": time_of_day}]
    )
    proba = pipeline.predict_proba(row)[0]
    model_classes = pipeline.named_steps["model"].classes_
    ranked = sorted(zip(model_classes, proba), key=lambda x: x[1], reverse=True)

    return jsonify(
        {
            "prediction": ranked[0][0],
            "top_k": top_k,
            "probabilities": [
                {
                    "coffee_name": name,
                    "probability": round(float(p), 4),
                    "in_top_k": rank < top_k,
                }
                for rank, (name, p) in enumerate(ranked)
            ],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
