from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
import io
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- Change this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load model and processor
repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
image_processor = AutoImageProcessor.from_pretrained(repo_name)
model = AutoModelForImageClassification.from_pretrained(repo_name)

# Class names
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

# Prediction function
def predict_skin_disease_from_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        encoding = image_processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class_name = class_names[predicted_class_idx]
        return predicted_class_name
    except Exception as e:
        return str(e)

# Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prediction = predict_skin_disease_from_bytes(contents)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
