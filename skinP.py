# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from PIL import Image
# import torch
# from transformers import AutoModelForImageClassification, AutoImageProcessor
# import io
# from fastapi.middleware.cors import CORSMiddleware

# # Initialize FastAPI app
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # <-- Change this in production!
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # Load model and processor
# repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
# image_processor = AutoImageProcessor.from_pretrained(repo_name)
# model = AutoModelForImageClassification.from_pretrained(repo_name)

# # Class names
# class_names = [
#     'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa', 'Hailey-Hailey Disease',
#     'Herpes Simplex', 'Impetigo', 'Larva Migrans', 'Leprosy Borderline', 'Leprosy Lepromatous',
#     'Leprosy Tuberculoid', 'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
#     'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
#     'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis', 'Pityriasis Rosea',
#     'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis', 'Tinea Nigra', 'Tungiasis',
#     'actinic keratosis', 'dermatofibroma', 'nevus', 'pigmented benign keratosis',
#     'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
# ]

# # Prediction function
# def predict_skin_disease_from_bytes(image_bytes):
#     try:
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         encoding = image_processor(image, return_tensors="pt")
#         with torch.no_grad():
#             outputs = model(**encoding)
#             logits = outputs.logits
#         predicted_class_idx = logits.argmax(-1).item()
#         predicted_class_name = class_names[predicted_class_idx]
#         return predicted_class_name
#     except Exception as e:
#         return str(e)

# # Endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         prediction = predict_skin_disease_from_bytes(contents)
#         return JSONResponse(content={"prediction": prediction})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import os

# --- Step 1: Initialize model and processor as None ---
# They will be loaded on the first request, not on startup.
app = FastAPI()
model = None
image_processor = None
model_path = "/tmp/model" # Vercel has a writable /tmp directory

# --- CORS Configuration (Your existing code is fine) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 2: Create a function to load the model ---
def load_model():
    """Checks if model is loaded, if not, downloads and loads it."""
    global model, image_processor
    # This import is heavy, so we only do it inside the function
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    if model is None:
        # If model exists in the temporary directory, load it from there
        if os.path.exists(model_path):
            print("Loading model from /tmp...")
            image_processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)
        # If not, download from Hugging Face and save to /tmp
        else:
            print("Downloading model from Hugging Face and saving to /tmp...")
            repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
            image_processor = AutoImageProcessor.from_pretrained(repo_name)
            model = AutoModelForImageClassification.from_pretrained(repo_name)
            # Save the model to the /tmp directory for future reuse
            image_processor.save_pretrained(model_path)
            model.save_pretrained(model_path)

# Your class names remain the same
class_names = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa', 'Hailey-Hailey Disease',
    'Herpes Simplex', 'Impetigo', 'Larva Migrans', 'Leprosy Borderline', 'Leprosy Lepromatous',
    'Leprosy Tuberculoid', 'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
    'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
    'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis', 'Pityriasis Rosea',
    'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis', 'Tinea Nigra', 'Tungiasis',
    'actinic keratosis', 'dermatofibroma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

# Prediction function is almost the same
def predict_skin_disease_from_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        encoding = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return class_names[predicted_class_idx]
    except Exception as e:
        return str(e)

# --- Step 3: Update your endpoint to call the load_model function ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load the model if it's not already loaded
    load_model()
    
    try:
        contents = await file.read()
        prediction = predict_skin_disease_from_bytes(contents)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
