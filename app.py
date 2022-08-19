from fastapi import FastAPI
import uvicorn
from fastapi import UploadFile, File
from utils import normalize_imgs
import numpy as np
from tensorflow.keras.models import load_model


app = FastAPI()
model = load_model('best_model')
labels = ['bike','motorcycle']


@app.get("/info")
async def get_info():
    return "This API retrieves if the image you import is a bike or a motorcycle"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image_array = normalize_imgs(bytes, (80,80), 1)
    #With this you can predict a single image without receiving an error
    image_array = np.expand_dims(image_array, axis = 0)
    prediction = model.predict(image_array)
    prediction_idx = prediction.argmax()
    confidence = round(float(np.max(prediction)), 2)

    return {
        "Prediction": labels[prediction_idx],
        "Confidence": confidence
        }


if __name__ == '__main__':
    uvicorn.run(app, port = 8082, host = 'localhost')
