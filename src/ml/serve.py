from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

MODEL_PATH = "artifacts/churn_model.keras"

app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    """
    Expected JSON:
    {
      "monthly_spend": 80,
      "support_tickets": 3,
      "contract_months": 6,
      "logins_per_week": 2
    }
    """
    body = request.get_json(force=True)
    x = np.array(
        [[
            float(body["monthly_spend"]),
            float(body["support_tickets"]),
            float(body["contract_months"]),
            float(body["logins_per_week"]),
        ]],
        dtype=np.float32,
    )

    prob = float(model.predict(x, verbose=0)[0][0])
    return jsonify(
        {
            "churn_probability": round(prob, 4),
            "risk_level": "high" if prob >= 0.6 else "low",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)