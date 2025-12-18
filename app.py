from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import fit  # Import the module to run main() if needed

app = Flask(__name__)
app.secret_key = "wine_research_super_secret"

MODEL_PATH = "models/model.pkl"
COLUMNS_PATH = "models/columns.pkl"

def load_artifacts():
    """Loads the model and columns. If missing, runs fit.main() to regenerate them."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        print("🚨 Artifacts not found! Initiating self-healing protocol... 🛠️")
        print("Running fit.main() to train and save models...")
        try:
            fit.main()
            print("✅ Self-healing complete. Artifacts regenerated.")
        except Exception as e:
            print(f"❌ Critical Error during self-healing: {e}")
            raise RuntimeError("Failed to regenerate artifacts.") from e
    
    try:
        model = joblib.load(MODEL_PATH)
        all_cols = joblib.load(COLUMNS_PATH)
        # fit.py saves all columns including the target 'label_quality' at the end.
        # We need to exclude the target column to get just the features.
        # x = df.iloc[:,:-1], so features are all_cols[:-1]
        columns = all_cols[:-1] if len(all_cols) > 0 else all_cols
        
        print(f"✨ Artifacts loaded. Features: {columns}")
        return model, columns
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        raise

# Load artifacts on startup
model, all_columns = load_artifacts()

@app.route('/')
def index():
    # Convert columns to a list for the template
    # all_columns is likely a pd.Index, convert to list
    features = all_columns.tolist() if hasattr(all_columns, 'tolist') else list(all_columns)
    # Filter out potential target columns if they accidentally slipped in, though fit.py seems to handle this.
    # fit.py: x = df.iloc[:,:-1], so all_columns has the features.
    
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Ensure we have all necessary columns
        input_data = {col: data.get(col) for col in all_columns}
        
        # Check for missing values (simple check)
        if any(v is None for v in input_data.values()):
            return jsonify({'error': 'Missing values for some features'}), 400

        # Create DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Predict
        # Convert to numpy to avoid sklearn warning about feature names mismatch
        # since the model was trained on numpy arrays (implied by fit.py)
        prediction = model.predict(df_input.to_numpy())[0]
        
        # Map prediction to maps
        # 0: Low, 1: Average, 2: High (Based on fit.py mapper)
        label_map = {0: "Low", 1: "Average", 2: "High"}
        result = label_map.get(prediction, "Unknown")
        
        return jsonify({'prediction': result, 'class_id': int(prediction)})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'columns_loaded': all_columns is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
