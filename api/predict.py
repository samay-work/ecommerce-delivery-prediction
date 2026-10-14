from http.server import BaseHTTPRequestHandler
import json
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "delivery_prediction_model.pkl"

package = joblib.load(MODEL_PATH)
model = package["model"]
features = package["features"]


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))

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

            result = {
                "prediction": prediction,
                "confidence": round(float(max(probabilities)) * 100, 2)
            }

            self.send_response(200)

        except Exception as e:
            result = {"error": str(e)}
            self.send_response(400)

        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
