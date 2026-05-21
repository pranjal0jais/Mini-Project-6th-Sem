from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from Phase_1_Classifier.phase1_predictor import PneumoniaPredictor
from Phase_2_Classifier.phase2_predictor import BacterialViralPredictor

app = FastAPI(title="Pneumonia Detection API")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models once at startup
phase1 = PneumoniaPredictor(
    model_path="model_repository/pneumonia_model_phase_1.pth",
    threshold=0.7904277
)
phase2 = BacterialViralPredictor(
    model_path="model_repository/pneumonia_model_phase_2.pth",
    threshold=0.4177126
)

def load_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_full(file: UploadFile = File(...)):
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
        "type_confidence": phase2_result["confidence"],
    })

@app.post("/predict/gradcam")
async def predict_full(file: UploadFile = File(...)):

    img = load_image(await file.read())
    phase1_result = phase1.predict(img)
    heatmap = phase1.generate_gradcam(img)

    if phase1_result["label"] == "NORMAL":
        return JSONResponse({
            "diagnosis": "NORMAL",
            "pneumonia_confidence": phase1_result["confidence"],
            "type": None,
            "type_confidence": None,
            "gradcam": heatmap
        })

    phase2_result = phase2.predict(img)
    return JSONResponse({
        "diagnosis": "PNEUMONIA",
        "pneumonia_confidence": phase1_result["confidence"],
        "type": phase2_result["label"],
        "type_confidence": phase2_result["confidence"],
        "gradcam": heatmap
    })

@app.get("/health")
def health():
    return {"status": "ok"}
