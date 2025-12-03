import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import onnxruntime as ort
import pickle
import json

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
# Load model + labels
# -----------------------------
ONNX_PATH = "mobilenetv3_birds.onnx"
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

import logging

# ------------ Robust labels loader ------------
def load_labels(path="labels.pkl"):
    """
    Loads labels.pkl (could be species->idx or idx->species or list).
    Returns idx_to_species dict: {0: "species name", 1: ...}
    """
    if not os.path.exists(path):
        raise RuntimeError(f"labels file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Case A: already idx -> species mapping (dict with integer keys)
    if isinstance(data, dict):
        # check keys
        # keys may be ints, strings of ints, or species names (string->int)
        keys = list(data.keys())
        # ints -> assume idx->species or species->idx (determine)
        if all(isinstance(k, int) for k in keys):
            # assume data is idx->species
            idx_to_species = {int(k): v for k, v in data.items()}
            logging.info("Loaded labels.pkl as idx->species mapping.")
            return idx_to_species

        # keys are strings: check whether values are ints (species->idx)
        if all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
            # invert species->idx to idx->species
            idx_to_species = {int(v): k for k, v in data.items()}
            logging.info("Loaded labels.pkl as species->idx mapping; inverted to idx->species.")
            return idx_to_species

        # keys are strings and values strings: maybe already idx->species but keys are str numbers
        if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            # try to convert keys to int where possible
            try:
                idx_to_species = {int(k): v for k, v in data.items()}
                logging.info("Loaded labels.pkl as str-int-key dict; converted keys to int.")
                return idx_to_species
            except Exception:
                pass

    # Case B: list or tuple => idx->species by position
    if isinstance(data, (list, tuple)):
        idx_to_species = {int(i): name for i, name in enumerate(data)}
        logging.info("Loaded labels.pkl as list/tuple; converted to idx->species.")
        return idx_to_species

    # Unknown format: raise helpful error
    raise RuntimeError(f"Unrecognized labels.pkl format: {type(data)}. Please supply species->idx dict or list of species.")

# Load and normalize
idx_to_species = load_labels("labels.pkl")
# To preserve prior code where you used LABELS[i], set LABELS variable to idx_to_species
LABELS = idx_to_species
NUM_CLASSES = len(LABELS)
print(f"Loaded {NUM_CLASSES} classes from labels.pkl (keys: {list(LABELS.keys())[:10]}...)")
# ------------ end loader ------------

# -----------------------------
# API Key validation
# -----------------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="HEY API KEY NYA BANG")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr

# -----------------------------
# NumPy softmax
# -----------------------------
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1, keepdims=True)

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3, x_api_key: str = Header(None)):
    verify_api_key(x_api_key)

    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image")

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

        results.append({
            "species": species,
            "score": float(probs[i]),
            "iucn": status
        })

    return {
        "top1": results[0],
        "topk": results
    }



@app.get("/")
def root():
    return {"status": "ok", "model": ONNX_PATH, "classes": len(LABELS)}
