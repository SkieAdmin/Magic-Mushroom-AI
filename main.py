import os
import base64
import time
import json
from typing import List
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from PIL import Image

from database import get_db, init_db
from models import Detection as DetectionModel
from schemas import DetectionResult, Detection, HistoryResponse, HistoryItem, PingResponse
from utils.freshness import classify_vegetable

load_dotenv()

app = FastAPI(
    title="Vegetable Freshness Scanner API",
    description="AI-powered vegetable freshness detection using YOLOv8",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Load YOLO model
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
model = YOLO(MODEL_PATH)


def get_gpu_name() -> str:
    """Detect GPU name or return CPU"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 JPEG string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)

        # Decode JPEG to OpenCV format
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {str(e)}")


def process_image(image: np.ndarray) -> dict:
    """
    Run YOLOv8 inference on image and return detection results
    in frontend-compatible format.

    Returns:
        {
            "detections": [
                {
                    "class": "rotten_carrot",
                    "confidence": 0.87,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": 1699999999999
                }
            ],
            "inferenceTime": 45.2,
            "fps": 0,  # Will be calculated by caller
            "gpuName": "NVIDIA RTX 3080"
        }
    """
    start_time = time.time()

    # Run YOLOv8 inference
    results = model(image, verbose=False)

    # Process results
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # Get class name and confidence
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])  # Keep 0-1 scale

                # Get bounding box coordinates [x1, y1, x2, y2]
                bbox = box.xyxy[0].tolist()

                # Classify vegetable (Healthy/Damaged/Rotten)
                classified_name, status, freshness_level, recommendation = classify_vegetable(
                    class_name, confidence
                )

                # Create detection object matching frontend interface
                detection = {
                    "class": classified_name,  # e.g., "rotten_carrot", "healthy_tomato"
                    "confidence": round(confidence, 4),  # 0-1 scale
                    "bbox": [round(coord, 2) for coord in bbox],
                    "timestamp": int(time.time() * 1000)
                }

                detections.append(detection)

    # Calculate inference time
    inference_time = (time.time() - start_time) * 1000

    # Return frontend-compatible structure
    return {
        "detections": detections,
        "inferenceTime": round(inference_time, 2),
        "fps": 0,  # Will be calculated in WebSocket handler
        "gpuName": get_gpu_name()
    }


def save_detection_to_db(db: Session, detection: dict):
    """Save detection to database for history"""
    try:
        # Extract first detection for storage
        if detection.get("detections") and len(detection["detections"]) > 0:
            first_det = detection["detections"][0]

            # Parse vegetable name and status from classified name
            classified_name = first_det["class"]
            confidence = first_det["confidence"]

            # Determine status and freshness
            if "rotten" in classified_name or "bad" in classified_name:
                status = "Bad"
                freshness = confidence * 40
                recommendation = "Not recommended for consumption."
            elif "damaged" in classified_name or "old" in classified_name:
                status = "Caution"
                freshness = 40 + (confidence * 35)
                recommendation = "Consume soon. Best for cooking."
            else:
                status = "Good"
                freshness = 70 + (confidence * 30)
                recommendation = "Recommended for consumption."

            # Clean vegetable name
            veg_name = classified_name.replace("rotten_", "").replace("damaged_", "").replace("healthy_", "")

            db_detection = DetectionModel(
                vegetable=veg_name,
                confidence=confidence * 100,  # Store as percentage in DB
                freshness=freshness,
                status=status,
                recommendation=recommendation
            )
            db.add(db_detection)
            db.commit()
            db.refresh(db_detection)
            return db_detection
    except Exception as e:
        print(f"Error saving to database: {e}")
        db.rollback()
    return None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print("🚀 Vegetable Freshness Scanner API v1.0.0")
    print("=" * 60)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🔌 Server: 0.0.0.0:9055")
    print(f"🎮 GPU: {get_gpu_name()}")
    print("=" * 60)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Vegetable Freshness Scanner API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "websocket": "/ws/detect",
            "upload": "/upload",
            "history": "/history",
            "ping": "/ping"
        }
    }


@app.get("/ping", response_model=PingResponse, tags=["Health"])
async def ping():
    """Health check endpoint"""
    start = time.time()
    latency_ms = round((time.time() - start) * 1000, 2)
    return PingResponse(status="online", latency_ms=latency_ms)


@app.post("/upload", response_model=DetectionResult, tags=["Detection"])
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Handle multipart image upload and return detection results.
    Returns frontend-compatible DetectionResult format.
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )

        # Process image
        result = process_image(image)

        # Save to database
        save_detection_to_db(db, result)

        # Return DetectionResult
        return DetectionResult(**result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}"}
        )


@app.get("/history", response_model=HistoryResponse, tags=["History"])
async def get_history(limit: int = 20, db: Session = Depends(get_db)):
    """Get detection history from database"""
    detections = db.query(DetectionModel).order_by(
        DetectionModel.timestamp.desc()
    ).limit(limit).all()

    history_items = [
        HistoryItem(
            id=d.id,
            timestamp=d.timestamp,
            vegetable=d.vegetable,
            confidence=d.confidence,
            freshness=d.freshness,
            status=d.status,
            recommendation=d.recommendation
        )
        for d in detections
    ]

    return HistoryResponse(detections=history_items, total=len(history_items))


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection.

    Client sends:
        {"type": "frame", "image": "<base64>", "timestamp": 123}

    Server responds:
        {
            "detections": [...],
            "inferenceTime": 45.2,
            "fps": 12.5,
            "gpuName": "NVIDIA RTX 3080"
        }
    """
    await websocket.accept()
    print("✅ WebSocket client connected")

    from database import SessionLocal
    db = SessionLocal()
    frame_times = []

    try:
        while True:
            # Receive base64 encoded JPEG frame
            data = await websocket.receive_text()

            frame_start = time.time()

            try:
                # Parse JSON if needed
                try:
                    json_data = json.loads(data)
                    base64_image = json_data.get("image", data)
                except json.JSONDecodeError:
                    base64_image = data

                # Decode base64 image
                image = decode_base64_image(base64_image)

                # Process image (returns DetectionResult format)
                result = process_image(image)

                # Calculate FPS
                frame_times.append(time.time())
                frame_times = frame_times[-30:]  # Keep last 30 frames

                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                else:
                    fps = 0

                # Update FPS in result
                result["fps"] = round(fps, 2)
                result["inferenceTime"] = round((time.time() - frame_start) * 1000, 2)

                # Save to database (async, don't block)
                save_detection_to_db(db, result)

                # Send response as DetectionResult
                await websocket.send_json(result)

                # Log detection info
                det_count = len(result.get("detections", []))
                if det_count > 0:
                    classes = [d["class"] for d in result["detections"]]
                    print(f"📦 Detected {det_count} object(s): {classes}")

            except ValueError as e:
                # Image decoding error
                await websocket.send_json({
                    "detections": [],
                    "inferenceTime": 0,
                    "fps": 0,
                    "gpuName": get_gpu_name(),
                    "error": str(e)
                })
                print(f"❌ Image decoding error: {e}")

            except Exception as e:
                # Processing error
                await websocket.send_json({
                    "detections": [],
                    "inferenceTime": 0,
                    "fps": 0,
                    "gpuName": get_gpu_name(),
                    "error": f"Processing error: {str(e)}"
                })
                print(f"❌ Processing error: {e}")

    except WebSocketDisconnect:
        print("❌ WebSocket client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 9055))

    print("\n" + "=" * 60)
    print("🥕 Starting Vegetable Freshness Scanner Backend")
    print("=" * 60)

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )
