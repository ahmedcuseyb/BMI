import streamlit as st
import pandas as pd
import joblib


# Load Model and Scaler
MODEL_PATH = "BMI.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="BMI Prediction App", page_icon="🧍", layout="centered")

st.title("BMI Prediction App")
st.write("Enter your details to predict the Your BMI ")

# =========================
# User Input Section
# =========================
st.subheader("Enter New Data")

num_rows = st.number_input("How many people do you want to test?", min_value=1, max_value=10, value=1)

input_data = []
for i in range(num_rows):
    st.markdown(f"### Person {i+1}")
    gender = st.selectbox(f"Gender (Person {i+1})", ["Male", "Female"], key=f"gender_{i}")
    height = st.number_input(f"Height (cm) (Person {i+1})", min_value=100, max_value=250, value=170, key=f"height_{i}")
    weight = st.number_input(f"Weight (kg) (Person {i+1})", min_value=30, max_value=200, value=70, key=f"weight_{i}")
    input_data.append({"Gender": gender, "Height": height, "Weight": weight})


# Predict Button
if st.button("Predict BMI Category"):
    # Convert to DataFrame (keep a copy of real data)
    real_data = pd.DataFrame(input_data)

    # ---------- Internal Preprocessing ----------
    encoded_data = real_data.copy()
    encoded_data["Gender"] = encoded_data["Gender"].replace({"Male": 0, "Female": 1})

    # Scale numeric columns (Height, Weight)
    encoded_data[["Height", "Weight"]] = scaler.transform(encoded_data[["Height", "Weight"]])

    # Predict
    predictions = model.predict(encoded_data)

    # ---------- Display Results ----------
    real_data["Predicted BMI INDEX"] = predictions

    st.success("Prediction completed successfully!")
    st.write("### Results (with real values):")
    st.dataframe(real_data)

    # Download predictions
    csv = real_data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions (CSV)", csv, "bmi_predictions.csv", "text/csv")
