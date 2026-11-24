
import os
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(
    title="Kestrel Service Routing API",
    description="AI-based service request routing",
    version="1.0"
)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "routing_model.pkl"
)

model_data = None


def load_model():
    global model_data

    if MODEL_PATH.exists():
        model_data = joblib.load(MODEL_PATH)
        return True

    return False


class ServiceRequest(BaseModel):
    request_text: str
    product_family: str
    warranty_status: str
    channel: str
    source: str = "crm"


def prepare_text(data):
    return (
        f"{data.request_text} "
        f"product {data.product_family} "
        f"warranty {data.warranty_status} "
        f"channel {data.channel} "
        f"source {data.source}"
    ).lower()


@app.get("/")
def home():
    return {"message": "Kestrel Routing API is running"}


@app.post("/predict")
def predict(data: ServiceRequest):

    if model_data is None:
        if not load_model():
            raise HTTPException(
                status_code=503,
                detail="Model unavailable. Run src/train.py first."
            )

    text = prepare_text(data)

    vectorizer = model_data["vectorizer"]
    model = model_data["model"]

    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]

    reasons = [
        f"Customer request: {data.request_text}",
        f"Product category: {data.product_family}",
        f"Warranty status: {data.warranty_status}",
        f"Request channel: {data.channel}"
    ]

    return {
        "predicted_team": prediction,
        "reasons": reasons
    }
