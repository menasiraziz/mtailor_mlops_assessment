from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import base64

app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

def load_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)
    return img

class OnnxClassifier:
    def __init__(self, onnx_path, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array):
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        return outputs[0]

    def predict_class(self, input_array):
        logits = self.predict(input_array)
        return int(np.argmax(logits, axis=1)[0])

classifier = OnnxClassifier("classifier_with_preprocessing.onnx")

@app.post("/predict")
def predict(req: ImageRequest):
    
    if not req.image or not req.image.strip():
        raise HTTPException(status_code=400, detail="Image field is empty or missing.")

    try:
        image_bytes = base64.b64decode(req.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Image is not valid base64.")

    try:
        img = load_image_from_bytes(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Decoded data is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")

    try:
        pred = classifier.predict_class(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    return { "class_id": pred}

@app.get("/health")
def health():
    return {"status": "ok"}