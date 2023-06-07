"""
Created on Wed June 7 13:27:41 2023
@author: win10
"""
# 1. Library imports
import base64
from fastapi import Body
import uvicorn
from fastapi import FastAPI, Request, Form
import numpy as np
from tensorflow import keras
from imagemodel import ImageModel
from fastapi.responses import JSONResponse

app = FastAPI()

# 2. Load the trained model
model_path = "deployment/Product Anomaly Model Deployment/productanomaly.h5"
model = keras.models.load_model(model_path)

# 3. Define the image transformation
input_shape = model.input_shape[1:3]
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )

@app.get("/predict")
def predict_species(data:ImageModel):
    data = data.dict()
    img = data['image']
    decoded_image = base64.b64decode(img)
    predictions = model.predict(decoded_image)
    predicted_class = np.argmax(predictions)
    probabilities = predictions
    pct = np.max(predictions)
    # perform further processing or save the image as needed
    return {"class": predicted_class, "probabilities": probabilities.tolist(), "pct":pct}

# 4. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 

# uvicorn app:app --host 0.0.0.0 --port 8000
# uvicorn app:app --reload

# http://127.0.0.1:8000/predict?img=aojghqwogiqhwgqwgiqwhgoqhgohqwgiogiqhwgio