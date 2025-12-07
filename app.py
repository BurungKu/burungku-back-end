import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from PIL import Image
import onnxruntime as ort
import pickle
import json
import joblib
import librosa
import logging

# -----------------------------
# Load API_KEY from Heroku env
# -----------------------------
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API KEY NYA JANGAN LUPA LHO YA")

# -----------------------------
# FastAPI init + CORS
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Standardized error system
# -----------------------------
class ApiException(HTTPException):
    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(
            status_code=status_code,
            detail={
                "success": False,
                "error_code": error_code,
                "message": message
            }
        )

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc: HTTPException):
    # If detail already formatted, return as is
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    # Else wrap it
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": "HTTP_ERROR",
            "message": str(exc.detail)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": exc.errors()
        }
    )

# -----------------------------
# Load labels + model
# -----------------------------
def load_labels(path="labels.pkl"):
    if not os.path.exists(path):
        raise RuntimeError(f"labels file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        keys = list(data.keys())
        if all(isinstance(k, int) for k in keys):
            return data
        if all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
            return {v: k for k, v in data.items()}
        if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            try:
                return {int(k): v for k, v in data.items()}
            except:
                pass

    if isinstance(data, (list, tuple)):
        return {i: name for i, name in enumerate(data)}

    raise RuntimeError("Unrecognized labels.pkl format")

idx_to_species = load_labels("labels.pkl")
LABELS = idx_to_species

session = ort.InferenceSession("mobilenetv3_birds.onnx", providers=["CPUExecutionProvider"])
AUDIO_MODEL = joblib.load("gb_audio_classifier.pkl")
SPECIES_ENCODER = joblib.load("species_encoder.pkl")
TYPE_ENCODER = joblib.load("type_encoder.pkl")
MAX_LEN = joblib.load("max_len.pkl")

# -----------------------------
# API Key validation
# -----------------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise ApiException(401, "INVALID_API_KEY", "HEY API KEY NYA BANG")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 4, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        raise ApiException(400, "INVALID_IMAGE", "Uploaded file is not a valid image")

    inp = preprocess(img)
    logits = session.run(None, {"input": inp})[0][0]
    probs = softmax(logits)
    idxs = probs.argsort()[::-1][:topk]

    with open("iucn_lookup_placeholder.json", "r") as f:
        IUCN = json.load(f)

    results = []
    for i in idxs:
        species = LABELS[i]
        status = IUCN.get(species, "Unknown")
        percentage = round(float(probs[i]) * 100, 2)

        results.append({
            "species": species,
            "score": percentage,
            "iucn": status
        })

    return {
        "success": True,
        "top1": results[0],
        "topk": results[1:4]
    }

# -----------------------------
# Audio prediction
# -----------------------------
@app.post("/predict_audio")
async def predict_audio(
    audio_file: UploadFile = File(...),
    species: str = Form(...),
    x_api_key: str = Header(None)
):
    verify_api_key(x_api_key)

    try:
        file_bytes = await audio_file.read()
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, duration=3)

        spec = librosa.feature.melspectrogram(y=y, sr=sr)
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = spec.T
        if spec.shape[0] < MAX_LEN:
            spec = np.pad(spec, ((0, MAX_LEN - spec.shape[0]), (0, 0)), mode="constant")
        else:
            spec = spec[:MAX_LEN]

        X_audio = spec.reshape(1, -1)
    except Exception as e:
        raise ApiException(400, "AUDIO_PROCESSING_FAILED", str(e))

    try:
        sp_idx = SPECIES_ENCODER.transform([species])[0]
    except:
        raise ApiException(400, "UNKNOWN_SPECIES", f"Unknown species: {species}")

    sp_onehot = np.eye(len(SPECIES_ENCODER.classes_))[sp_idx].reshape(1, -1)
    X = np.hstack([X_audio, sp_onehot])

    try:
        pred = AUDIO_MODEL.predict(X)[0]
        pred_label = TYPE_ENCODER.inverse_transform([pred])[0]
    except Exception as e:
        raise ApiException(500, "AUDIO_MODEL_ERROR", str(e))

    return {
        "success": True,
        "species": species,
        "predicted_simple_type": pred_label
    }

# -----------------------------
# Root
# -----------------------------
@app.get("/")
def root():
    return {"success": True, "status": "ok", "classes": len(LABELS)}
