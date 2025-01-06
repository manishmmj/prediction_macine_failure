import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

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

# Streamlit UI components
st.title("Maintenance Prediction")

# Input fields for the user
air_temp = st.number_input("Air temperature [K]", min_value=-100.0, value=25.0, step=0.1)
process_temp = st.number_input("Process temperature [K]", min_value=-100.0, value=50.0, step=0.1)
rotational_speed = st.number_input("Rotational speed [rpm]", min_value=0.0, value=3000.0, step=1.0)
torque = st.number_input("Torque [Nm]", min_value=0.0, value=150.0, step=1.0)
tool_wear = st.number_input("Tool wear [min]", min_value=0.0, value=100.0, step=1.0)
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

# Button to trigger prediction
if st.button("Predict"):
    try:
        # Encode the machine type
        machine_type_encoded = type_mapping[machine_type]

        # Create feature array
        features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear, machine_type_encoded]])

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] if hasattr(model, "predict_proba") else None

        # Display result
        st.subheader("Prediction Results:")
        st.write(f"Prediction: {'Failure' if prediction == 1 else 'No Failure'}")
        st.write(f"Probability of Failure: {round(probability, 4) if probability else 'N/A'}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")


