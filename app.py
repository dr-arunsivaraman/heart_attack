from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText

app = FastAPI()

# Load model once
model = joblib.load("model.pkl")

def send_email(to_email: str, subject: str, body: str):
    sender = "rrawspractice@gmail.com"
    password = "kkdrkwqxegzffjbt"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())

class HeartInput(BaseModel):
    Age: float
    Gender: int
    Heart_rate: float
    Systolic_blood_pressure: float
    Diastolic_blood_pressure: float
    Blood_sugar: float
    CK_MB: float
    Troponin: float

@app.post("/predict")
def predict(data: HeartInput, background_tasks: BackgroundTasks):
    x = np.array([[
        data.Age,
        data.Gender,
        data.Heart_rate,
        data.Systolic_blood_pressure,
        data.Diastolic_blood_pressure,
        data.Blood_sugar,
        data.CK_MB,
        data.Troponin
    ]])

    pred = model.predict(x)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(x)))

    # ✅ Response for UI (fast)
    result = {"prediction": str(pred), "confidence": confidence}

    # ✅ Email to Zapier (background - no timeout)
    zapier_email = "rithick.rcmkq5@zapiermail.com"
    if zapier_email:
        email_body = (
            f"Heart attack prediction result\n\n"
            f"prediction: {result['prediction']}\n"
            f"confidence: {result['confidence']}\n"
            f"Age: {data.Age}, Gender: {data.Gender}, Heart_rate: {data.Heart_rate}\n"
            f"SBP: {data.Systolic_blood_pressure}, DBP: {data.Diastolic_blood_pressure}\n"
            f"Blood_sugar: {data.Blood_sugar}, CK_MB: {data.CK_MB}, Troponin: {data.Troponin}\n"
        )

        background_tasks.add_task(
            send_email,
            zapier_email,
            "Heart attack prediction",
            email_body
        )

    return result

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
