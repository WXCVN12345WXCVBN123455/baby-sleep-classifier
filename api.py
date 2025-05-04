import os
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from PIL import Image
import uvicorn

app = FastAPI()

# Mount templates and static directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the model
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('best_model.h5')

@app.on_event("startup")
async def startup_event():
    load_model()

def preprocess_image(image):
    # Resize image to match model's expected sizing
    img = image.resize((224, 224))
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    probability = float(prediction[0][0])
    
    # Convert probability to class label
    class_label = "Belly" if probability > 0.5 else "Not Belly"
    confidence = probability if probability > 0.5 else 1 - probability
    
    return {
        "class": class_label,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000))) 