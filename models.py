from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    vegetable = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    freshness = Column(Float, nullable=False)
    status = Column(String, nullable=False)  # "Good", "Caution", "Bad"
    recommendation = Column(String, nullable=True)

