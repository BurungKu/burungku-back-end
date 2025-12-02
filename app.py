from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime as ort
import io

from utils import preprocess, postprocess

app = FastAPI()

session = ort.InferenceSession("yolov9t.onnx", providers=["CPUExecutionProvider"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_input, orig_w, orig_h = preprocess(img)

    outputs = session.run(None, {"input": img_input})

    detections = postprocess(outputs, orig_w, orig_h)

    return JSONResponse({"detections": detections})
