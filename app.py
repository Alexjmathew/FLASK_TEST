"""Simple Flask app that predicts house prices with a linear regression model."""
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

MODEL_PATH = "model.pkl"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]


def make_demo_data(n=1000, seed=42):
    """Generate synthetic housing data. Replace with pd.read_csv('your_data.csv')."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 4000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 4, n),
        "age_years": rng.integers(0, 60, n),
    })
    df["price"] = (
        50_000
        + df["area_sqft"] * 120
        + df["bedrooms"] * 8_000
        + df["bathrooms"] * 12_000
        - df["age_years"] * 900
        + rng.normal(0, 15_000, n)
    )
    return df


def train_model():
    df = make_demo_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["price"], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    print(f"Model trained. R^2 on test set: {r2_score(y_test, model.predict(X_test)):.3f}")
    joblib.dump(model, MODEL_PATH)
    return model


model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train_model()


def parse_inputs(data):
    """Validate and convert inputs to a one-row DataFrame."""
    values = {}
    for f in FEATURES:
        try:
            v = float(data[f])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"'{f}' is required and must be a number.")
        if v < 0:
            raise ValueError(f"'{f}' cannot be negative.")
        values[f] = v
    return pd.DataFrame([values], columns=FEATURES)


@app.route("/")
def index():
    return render_template("index.html", prediction=None, error=None, form={})


@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        price = float(model.predict(parse_inputs(request.form))[0])
        return render_template("index.html", prediction=f"{max(price, 0):,.2f}",
                               error=None, form=request.form)
    except ValueError as e:
        return render_template("index.html", prediction=None, error=str(e), form=request.form)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        price = float(model.predict(parse_inputs(request.get_json(silent=True) or {}))[0])
        return jsonify({"predicted_price": round(max(price, 0), 2)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
