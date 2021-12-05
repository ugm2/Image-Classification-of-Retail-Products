"""Retail Classification Service."""
from fastapi import FastAPI, UploadFile, File
from retail_multi_model.api.model import RetailResponse
from retail_multi_model.core.model import ViTForImageClassification
from PIL import Image

app = FastAPI()

model = None

def init_model(model_path):
    """Initialize model if not already initialized."""
    global model
    if model is None:
        model = ViTForImageClassification('google/vit-base-patch16-224')
        model.load(model_path)

    return model

@app.post("/classify",
    response_model=RetailResponse,
    status_code=200,
    name="classify_retail_items")
async def classify_retail_items(file: UploadFile = File(...)):
    """Classify retail items."""
    model = init_model("/home/unai/personal/Image-Classification-of-Retail-Products/model")
    image = Image.open(file.file)
    prediction = model.predict(image)
    print(prediction)
    return RetailResponse(product_prediction=prediction[0])