from fastapi import FastAPI
from pydantic import BaseModel, Field ,field_validator
from typing import Literal
import pickle
import numpy as np
from datetime import date
from calendar import monthrange

with open("delivery_time_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Food Delivery Time Prediction API")

def month_end_day(dt: date):
    return monthrange(dt.year, dt.month)[1]

class PredictionInput(BaseModel):
    age: float = Field(..., gt=0, lt=100)
    rating: float = Field(..., ge=0, le=5)
    distance: float = Field(..., gt=0)
    festival: bool
    weather_condition: Literal['Sunny', 'Stormy', 'Cloudy', 'Fog', 'Windy']
    delivery_date: date = Field(..., description="Delivery Date (YYYY-MM-DD)", example="2025-12-11")
    
    @field_validator('weather_condition', mode='before')
    def clean_weather(cls, v):
        return v.strip().capitalize()
    
@app.get("/")
def root():
    return {"message": "Welcome to the Delivery Time Prediction API"}

@app.post("/predict")
def predict_delivery_time(input_data: PredictionInput):
    delivery_date = input_data.delivery_date

    features = np.zeros((1, 17))
    features[0, 0] = input_data.age
    features[0, 1] = input_data.rating
    features[0, 4] = input_data.distance
    features[0, 2] = int(input_data.festival)

    weather_map = {'Sunny': 0, 'Stormy': 1, 'Cloudy': 2, 'Fog': 3, 'Windy': 4}
    features[0, 3] = weather_map[input_data.weather_condition]

    features[0, 5] = delivery_date.day
    features[0, 6] = delivery_date.month
    features[0, 7] = (delivery_date.month - 1) // 3 + 1
    features[0, 8] = delivery_date.year
    features[0, 9] = delivery_date.weekday()
    features[0, 10] = 1 if delivery_date.day == 1 else 0
    features[0, 11] = 1 if delivery_date.day == month_end_day(delivery_date) else 0
    features[0, 12] = 1 if delivery_date.month in [1, 4, 7, 10] and delivery_date.day == 1 else 0
    features[0, 13] = 1 if delivery_date.month in [3, 6, 9, 12] and delivery_date.day == month_end_day(delivery_date) else 0
    features[0, 14] = 1 if delivery_date.month == 1 and delivery_date.day == 1 else 0
    features[0, 15] = 1 if delivery_date.month == 12 and delivery_date.day == 31 else 0
    features[0, 16] = 1 if delivery_date.weekday() >= 5 else 0

    features_scaled = scaler.transform(features)
    predicted_time = model.predict(features_scaled)[0]

    return {"predicted_delivery_time_minutes": round(float(predicted_time), 2)}
