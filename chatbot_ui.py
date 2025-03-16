import streamlit as st
import requests

# FastAPI backend URL (Update this after deployment)
API_URL = "http://127.0.0.1:8000/predict"

# Streamlit UI
st.set_page_config(page_title="Healthcare Chatbot", layout="centered")

st.title("Healthcare Chatbot")
st.write("Enter your symptoms to predict the disease.")

# User input
user_input = st.text_area("Your Symptoms (comma-separated)", "")

# Predict button
if st.button("Predict Disease"):
    if user_input:
        response = requests.post(API_URL, json={"symptoms": user_input})
        if response.status_code == 200:
            predicted_disease = response.json().get("Predicted Disease", "Unknown")
            st.success(f" Predicted Disease: **{predicted_disease}**")
        else:
            st.error("Error connecting to the prediction server.")
    else:
        st.warning("Please enter symptoms before predicting.")

st.caption("🔬 Powered by BioBERT ")
