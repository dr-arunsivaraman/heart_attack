from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText

app = FastAPI()
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
def predict(data: HeartInput):
    x = np.array([[
        data.Age, data.Gender, data.Heart_rate,
        data.Systolic_blood_pressure, data.Diastolic_blood_pressure,
        data.Blood_sugar, data.CK_MB, data.Troponin
    ]])

    pred = model.predict(x)[0]
    confidence = float(np.max(model.predict_proba(x))) if hasattr(model, "predict_proba") else None

    # ✅ Send email (optional)
    zapier_email = "rithick.rcmkq5@zapiermail.com"
    if zapier_email:
        body = f"prediction: {pred}\nconfidence: {confidence}"
        send_email(zapier_email, "Heart attack prediction", body)

    return {"prediction": str(pred), "confidence": confidence}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
