import streamlit as st
import requests
from datetime import date

st.title("🚚 Food Delivery Time Prediction")

# User input fields
age = st.number_input("Age of Delivery Partner", min_value=0, max_value=100)
rating = st.slider("Average Rating", 0.0, 5.0, 0.1, step=0.1)
distance = st.number_input("Total Distance (in km)", min_value=0.0)
festival = st.checkbox("Is there a festival?")
weather_condition = st.selectbox("Weather Condition", ['Sunny', 'Stormy', 'Cloudy', 'Fog', 'Windy'])
delivery_date = st.date_input("Delivery Date", value=date.today())

if st.button("Predict Delivery Time"):
    url = "http://127.0.0.1:8000/predict"

    payload = {
        "age": age,
        "rating": rating,
        "distance": distance,
        "festival": festival,
        "weather_condition": weather_condition,
        "delivery_date": delivery_date.isoformat()  # convert to 'YYYY-MM-DD'
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Delivery Time: {result['predicted_delivery_time_minutes']} minutes ⏱️")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
