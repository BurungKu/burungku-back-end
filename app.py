from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import os

from utils import preprocess, postprocess

app = FastAPI()

API_KEY = os.getenv("API_KEY") 


session = None
def get_session():
    global session
    if session is None:
        session = ort.InferenceSession("yolov9t.onnx", providers=["CPUExecutionProvider"])
    return session

def verify_api_key(x_api_key: str | None):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API KEY TOLONG DIKONDISIKAN YH")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="HEYYYY API KEY")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(None)
):

    verify_api_key(x_api_key)

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_input, orig_w, orig_h = preprocess(img)

    ort_session = get_session()
    outputs = ort_session.run(None, {"images": img_input})

    detections = postprocess(outputs, orig_w, orig_h)

    return JSONResponse({"detections": detections})
