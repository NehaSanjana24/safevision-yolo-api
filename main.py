
import os
import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="SafeVision AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain for production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError("best.pt is missing from the repository.")
    model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "SafeVision AI API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=400, detail="Invalid confidence")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image")

    content = await file.read()

    if not content or len(content) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image must be non-empty and no larger than 15 MB"
        )

    frame = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = model.predict(
        source=frame,
        conf=confidence,
        verbose=False
    )[0]

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append({
                "class_id": class_id,
                "class_name": str(result.names[class_id]),
                "confidence": round(float(box.conf[0].item()), 4),
                "bbox": [
                    round(float(value))
                    for value in box.xyxy[0].tolist()
                ]
            })

    success, image = cv2.imencode(".jpg", result.plot())

    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    return {
        "success": True,
        "count": len(detections),
        "detections": detections,
        "annotated_image_base64": base64.b64encode(
            image.tobytes()
        ).decode("utf-8")
    }
