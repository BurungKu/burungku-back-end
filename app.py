from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime as ort
import io

from utils import preprocess, postprocess

app = FastAPI()

# Lazily load session to avoid Heroku startup crashes
session = None
def get_session():
    global session
    if session is None:
        session = ort.InferenceSession("yolov9t.onnx", providers=["CPUExecutionProvider"])
    return session

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_input, orig_w, orig_h = preprocess(img)

    ort_session = get_session()
    outputs = ort_session.run(None, {"input": img_input})

    detections = postprocess(outputs, orig_w, orig_h)

    return JSONResponse({"detections": detections})
