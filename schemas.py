from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

# Frontend-compatible schemas
class Detection(BaseModel):
    """Single detection matching frontend Detection interface"""
    class_name: str = Field(alias="class")
    confidence: float  # 0-1 scale
    bbox: List[float]  # [x1, y1, x2, y2]
    timestamp: int

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "class": "rotten_carrot",
                "confidence": 0.87,
                "bbox": [100, 150, 300, 400],
                "timestamp": 1699999999999
            }
        }

class DetectionResult(BaseModel):
    """Main detection result matching frontend DetectionResult interface"""
    detections: List[Detection]
    inferenceTime: float  # milliseconds
    fps: float
    gpuName: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "detections": [
                    {
                        "class": "rotten_carrot",
                        "confidence": 0.87,
                        "bbox": [100, 150, 300, 400],
                        "timestamp": 1699999999999
                    }
                ],
                "inferenceTime": 45.2,
                "fps": 12.5,
                "gpuName": "NVIDIA RTX 3080"
            }
        }

# Database history schemas
class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    vegetable: str
    confidence: float
    freshness: float
    status: str
    recommendation: Optional[str] = None

    class Config:
        from_attributes = True

class HistoryResponse(BaseModel):
    detections: List[HistoryItem]
    total: int

class PingResponse(BaseModel):
    status: str
    latency_ms: float

