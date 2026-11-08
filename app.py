from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Car Price Prediction")
model = joblib.load(Path(__file__).with_name("model.pkl"))


class CarDetails(BaseModel):
    age_years: int = Field(ge=0)
    mileage_km: int = Field(ge=0)
    engine_cc: int = Field(gt=0)


@app.post("/predict")
def predict(car: CarDetails):
    features = pd.DataFrame([car.model_dump()])
    price = model.predict(features)[0]
    return {"predicted_price_lakh": round(float(price), 2)}
