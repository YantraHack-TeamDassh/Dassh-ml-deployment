"""
Created on Wed June 7 13:27:41 2023
@author: win10
"""
# 1. Library imports
import base64
import io
import re
from fastapi import File, UploadFile
import uvicorn
from fastapi import FastAPI, Request
import numpy as np
from tensorflow import keras
from imagemodel import Prediction,ImageModel
from fastapi.responses import JSONResponse
from PIL import Image

main = FastAPI()

# 2. Load the trained model
model_path = "deployment/Product Anomaly Model Deployment/productanomaly.h5"
model = keras.models.load_model(model_path)

# 3. Define the image transformation
# input_shape = model.input_shape[1:3]
class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

@main.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

@main.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )

@main.post("/predict", response_model=Prediction)
async def prediction_route(img: ImageModel):
    # user_image = await image.read()
    img = img.dict()
    encoded = img['image']
    image_data = re.sub('^data:image/.+;base64,', '', encoded)
    base64bytes = base64.b64decode(image_data)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    img = img.resize((299,299))
    byteIO = io.BytesIO()
    img.save(byteIO, format='PNG')
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    predictions = model.predict(array)
    categories = ["not_ok","ok"]
    predicted_class = categories[np.argmax(predictions)]
    return {"predicted_class": predicted_class}
        
# 4. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(main, host='127.0.0.1', port=8000) 

# uvicorn main:main --host 0.0.0.0 --port 8000
# uvicorn main:main --reload

# http://127.0.0.1:8000/predict?img=aojghqwogiqhwgqwgiqwhgoqhgohqwgiogiqhwgio