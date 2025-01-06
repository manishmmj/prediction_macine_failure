from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load("best_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")  # Ensure the scaler is saved during training

# Define feature columns
feature_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type"
]

# Map for Type encoding
type_mapping = {"L": 0, "M": 1, "H": 2}

@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML file named index.html for the front-end

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the request
    data = request.json

    try:
        # Parse features
        air_temp = float(data["Air temperature [K]"])
        process_temp = float(data["Process temperature [K]"])
        rotational_speed = float(data["Rotational speed [rpm]"])
        torque = float(data["Torque [Nm]"])
        tool_wear = float(data["Tool wear [min]"])
        machine_type = type_mapping[data["Type"]]

        # Create feature array
        features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear, machine_type]])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else None

        # Create response
        response = {
            "prediction": int(prediction),
            "probability": round(probability, 4) if probability else "N/A"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
