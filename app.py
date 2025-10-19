from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load model and features
HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
MODEL_FILE = MODELS_DIR / "wine_pipeline.joblib"
FEATURE_FILE = MODELS_DIR / "wine_all_features.joblib"

pipeline = joblib.load(MODEL_FILE)
feature_names = joblib.load(FEATURE_FILE)
CLASS_MAP = {0: "Low quality (3-4)", 1: "Medium quality (5-6)", 2: "High quality (7-8)"}

@app.route("/")
def index():
    return render_template("index.html", features=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    values = []
    for f in feature_names:
      val = data.get(f, "")
      if val == "" or val is None:
          val = np.nan   # or 0.0 if you prefer
      else:
          val = float(val)
      values.append(val)

    X = np.array(values).reshape(1, -1)

    pred = pipeline.predict(X)[0]
    pred_proba = pipeline.predict_proba(X)[0].tolist()

    return jsonify({
        "predicted_class": int(pred),
        "predicted_label": CLASS_MAP[int(pred)],
        "probabilities": pred_proba
    })

if __name__ == "__main__":
    app.run(debug=True)
