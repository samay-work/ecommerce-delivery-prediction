
from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

model_path = Path(__file__).parent / "delivery_prediction_model.pkl"
package = joblib.load(model_path)

model = package["model"]
features = package["features"]


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        categorical = [
            "Warehouse_block",
            "Mode_of_Shipment",
            "Product_importance",
            "Gender"
        ]

        df = pd.DataFrame([data])

        df = pd.get_dummies(
            df,
            columns=categorical,
            drop_first=False,
            dtype=int
        )

        df = df.reindex(columns=features, fill_value=0)

        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]

        return jsonify({
            "prediction": prediction,
            "confidence": round(float(max(probabilities)) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run()
