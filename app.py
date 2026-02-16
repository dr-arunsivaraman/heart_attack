from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # run model prediction here
    return {"prediction": "Normal", "confidence": 0.92}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
