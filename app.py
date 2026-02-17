from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np

app = FastAPI()

# Load model once (put model.pkl in same folder)
model = joblib.load("model.pkl")

def send_email(to_email, subject, body):
    sender = "rrawspractice@gmail.com"
    password = "Rithick6040$"

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

    # if model supports probability
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(x)))

    return {"prediction": str(pred), "confidence": confidence} 
    zapier_email = "rithick.rcmkq5@zapiermail.com"
            email_body = f"""
            prediction: {prediction}
            confidence: {confidence}
            filename: {filename}
            """

            send_email(
                zapier_email,
                "Heart attack prediction",
                email_body
            )


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
