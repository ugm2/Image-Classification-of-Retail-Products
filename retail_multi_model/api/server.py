"""Retail Classification Service."""
from fastapi import FastAPI, UploadFile, File
from retail_multi_model.api.model import RetailResponse, RetailLabels
from retail_multi_model.core.model import ViTForImageClassification
from PIL import Image
import os

model_path = os.environ.get('MODEL_PATH', "./model")

app = FastAPI()

model = None

def init_model(model_path):
    """Initialize model if not already initialized."""
    global model
    if model is None:
        model = ViTForImageClassification('google/vit-base-patch16-224')
        model.load(model_path)

    return model

@app.post("/predict",
    response_model=RetailResponse,
    status_code=200,
    name="predict_retail_items")
async def classify_retail_items(image: UploadFile = File(...)):
    """Predict retail items."""
    model = init_model(model_path)
    image = Image.open(image.file)
    prediction, confidence = model.predict(image)
    return RetailResponse(prediction=prediction[0], confidence=round(confidence[0], 3))

@app.post("/get_labels",
    response_model=RetailLabels,
    status_code=200,
    name="get_retail_labels")
def get_retail_labels():
    """Get retail labels."""
    model = init_model(model_path)
    return RetailLabels(labels=list(model.label_encoder.classes_))