
import os
import cv2
import time
import tempfile
import threading
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO

app = FastAPI(title="SafeVision AI Image and Video API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend origin for production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 30
MAX_VIDEO_FRAMES = 900

model = None
model_lock = threading.Lock()


@app.on_event("startup")
def load_model():
    global model

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError("best.pt is missing from the repository")

    model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "SafeVision AI API is running",
        "endpoints": ["/health", "/predict", "/predict-video"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


def check_model():
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not loaded"
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0, le=1)
):
    check_model()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image")

    content = await file.read()

    if not content or len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be non-empty and no larger than 15 MB"
        )

    frame = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        with model_lock:
            result = model.predict(
                frame, conf=confidence, verbose=False
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
                        round(float(v))
                        for v in box.xyxy[0].tolist()
                    ]
                })

        success, encoded = cv2.imencode(".jpg", result.plot())

        if not success:
            raise HTTPException(
                status_code=500, detail="Image encoding failed"
            )

        import base64

        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "annotated_image_base64": base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {exc}"
        ) from exc


@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0, le=1)
):
    check_model()

    allowed_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    filename = (file.filename or "").lower()

    if not filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=415,
            detail="Upload MP4, AVI, MOV, MKV, or WebM video"
        )

    input_path = None
    output_path = None
    cap = None

    try:
        # Save upload in chunks and enforce the file-size limit.
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(filename)[1],
            delete=False
        ) as temp_input:
            input_path = temp_input.name
            total_bytes = 0

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 100 MB limit"
                    )

                temp_input.write(chunk)

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open this video"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if (
            width <= 0 or height <= 0 or fps <= 0
            or reported_frames <= 0
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid video metadata"
            )

        duration = reported_frames / fps

        if duration > MAX_VIDEO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail="Video duration limit is 30 seconds"
            )

        if reported_frames > MAX_VIDEO_FRAMES:
            raise HTTPException(
                status_code=413,
                detail="Video has too many frames; use a shorter or lower-FPS video"
            )

        output_file = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        )
        output_path = output_file.name
        output_file.close()

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Could not create output video"
            )

        class_counts = {}
        confidence_sums = {}
        frames_processed = 0
        total_detections = 0
        start_time = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                with model_lock:
                    result = model.predict(
                        frame,
                        conf=confidence,
                        verbose=False
                    )[0]

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = str(result.names[class_id])
                        score = float(box.conf[0].item())

                        class_counts[class_name] = (
                            class_counts.get(class_name, 0) + 1
                        )
                        confidence_sums[class_name] = (
                            confidence_sums.get(class_name, 0.0) + score
                        )
                        total_detections += 1

                writer.write(result.plot())
                frames_processed += 1

                # Safety guard against unexpectedly long processing.
                if time.time() - start_time > 240:
                    raise HTTPException(
                        status_code=408,
                        detail="Video processing timed out; try a shorter video"
                    )
        finally:
            cap.release()
            writer.release()

        if frames_processed == 0 or not os.path.exists(output_path):
            raise HTTPException(
                status_code=400,
                detail="No video frames could be processed"
            )

        summary = "; ".join(
            f"{name}: {count}"
            for name, count in sorted(class_counts.items())
        ) or "No objects detected"

        response_headers = {
            "X-Frames-Processed": str(frames_processed),
            "X-Detection-Count": str(total_detections),
            "X-Detection-Summary": summary[:1000]
        }

        # Delete the generated file after it has been sent.
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="safevision_annotated.mp4",
            headers=response_headers,
            background=BackgroundTask(
                lambda: os.path.exists(output_path)
                and os.remove(output_path)
            )
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video inference failed: {exc}"
        ) from exc
    finally:
        if cap is not None:
            cap.release()

        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        await file.close()

