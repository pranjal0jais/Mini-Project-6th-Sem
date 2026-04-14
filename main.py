from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from Phase_1_Classifier.phase1_predictor import PneumoniaPredictor
from Phase_2_Classifier.phase2_predictor import BacterialViralPredictor

app = FastAPI(title="Pneumonia Detection API")

# Load both models once at startup
phase1 = PneumoniaPredictor(
    model_path="model_repository/pneumonia_model_phase_1.pth",
    threshold=0.4
)
phase2 = BacterialViralPredictor(
    model_path="model_repository/pneumonia_model_phase_2.pth",
    threshold=0.53
)

def load_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")


@app.post("/predict/phase1")
async def predict_phase1(file: UploadFile = File(...)):
    """Predict: NORMAL vs PNEUMONIA"""
    img = load_image(await file.read())
    result = phase1.predict(img)
    return JSONResponse(result)


@app.post("/predict/phase2")
async def predict_phase2(file: UploadFile = File(...)):
    """Predict: BACTERIAL vs VIRAL (only for confirmed pneumonia cases)"""
    img = load_image(await file.read())
    result = phase2.predict(img)
    return JSONResponse(result)


@app.post("/predict/full")
async def predict_full(file: UploadFile = File(...)):
    """
    Full pipeline: first check for pneumonia,
    then classify type if pneumonia is detected.
    """
    img = load_image(await file.read())
    phase1_result = phase1.predict(img)

    if phase1_result["label"] == "NORMAL":
        return JSONResponse({
            "diagnosis": "NORMAL",
            "pneumonia_confidence": phase1_result["confidence"],
            "type": None,
            "type_confidence": None
        })

    phase2_result = phase2.predict(img)
    return JSONResponse({
        "diagnosis": "PNEUMONIA",
        "pneumonia_confidence": phase1_result["confidence"],
        "type": phase2_result["label"],
        "type_confidence": phase2_result["confidence"]
    })


@app.get("/health")
def health():
    return {"status": "ok"}
