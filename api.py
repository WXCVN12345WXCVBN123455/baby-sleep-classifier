from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os

app = FastAPI(title="Baby Sleep Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = tf.keras.models.load_model("baby_sleep_classifier.h5")

def preprocess_image(image_bytes):
    # Open and resize the image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(contents)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get the probability
        probability = float(prediction[0][0])
        
        # Determine the class
        class_name = "belly" if probability > 0.5 else "not_belly"
        
        return {
            "class": class_name,
            "probability": probability,
            "confidence": f"{probability:.2%}" if class_name == "belly" else f"{(1 - probability):.2%}"
        }
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 