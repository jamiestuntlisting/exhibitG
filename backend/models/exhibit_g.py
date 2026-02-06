from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Base


class ExhibitG(Base):
    """Represents one uploaded Exhibit G form."""
    __tablename__ = "exhibit_g"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(Text, nullable=False)
    original_image_path = Column(Text, nullable=False)
    corrected_image_path = Column(Text, nullable=True)
    skew_angle = Column(Float, nullable=True)

    # Header fields
    production_co = Column(String(255), default="")
    series_title = Column(String(255), default="")
    episode_name = Column(String(255), default="")
    prod_number = Column(String(100), default="")
    date = Column(String(50), default="")
    day_x_of_y = Column(String(50), default="")
    contact = Column(String(255), default="")
    phone = Column(String(100), default="")

    status = Column(String(20), default="processing")  # processing, review, complete
    created_at = Column(DateTime, default=datetime.utcnow)

    # Cell bounding boxes for the detected grid (JSON)
    cell_grid = Column(JSON, nullable=True)

    # Relationships
    rows = relationship("ExhibitGRow", back_populates="exhibit_g", cascade="all, delete-orphan",
                        order_by="ExhibitGRow.row_number")
    transcription_runs = relationship("TranscriptionRun", back_populates="exhibit_g", cascade="all, delete-orphan")
    disagreements = relationship("CellDisagreement", back_populates="exhibit_g", cascade="all, delete-orphan")


class ExhibitGRow(Base):
    """One row of the cast table."""
    __tablename__ = "exhibit_g_row"

    id = Column(Integer, primary_key=True, index=True)
    exhibit_g_id = Column(Integer, ForeignKey("exhibit_g.id"), nullable=False)
    row_number = Column(Integer, nullable=False)

    cast_number = Column(String(10), default="")
    cast_name = Column(String(255), default="")
    character = Column(String(255), default="")
    status_code = Column(String(10), default="")  # W, H, S, WF, etc.

    # Work time
    makeup_time = Column(String(20), default="")
    report_on_set = Column(String(20), default="")
    dismiss_on_set = Column(String(20), default="")
    dismiss_mu_hair = Column(String(20), default="")

    # Meals - ND Breakfast
    nd_breakfast_out = Column(String(20), default="")
    nd_breakfast_in = Column(String(20), default="")

    # 1st Meal
    first_meal_in = Column(String(20), default="")
    first_meal_out = Column(String(20), default="")

    # 2nd Meal
    second_meal_in = Column(String(20), default="")
    second_meal_out = Column(String(20), default="")

    # Travel time
    leave_for_loc = Column(String(20), default="")
    arrive_on_loc = Column(String(20), default="")
    leave_loc = Column(String(20), default="")
    arrive_hotel = Column(String(20), default="")

    # Other
    stunt_adjust = Column(String(50), default="")
    mileage = Column(String(50), default="")
    mpv_fc = Column(String(50), default="")

    exhibit_g = relationship("ExhibitG", back_populates="rows")


class TranscriptionRun(Base):
    """Tracks each OCR engine's reading of a specific cell."""
    __tablename__ = "transcription_run"

    id = Column(Integer, primary_key=True, index=True)
    exhibit_g_id = Column(Integer, ForeignKey("exhibit_g.id"), nullable=False)
    engine = Column(String(20), nullable=False)  # paddleocr, easyocr, claude
    row_number = Column(Integer, nullable=False)
    column_name = Column(String(50), nullable=False)
    value = Column(Text, default="")
    confidence = Column(Float, nullable=True)

    exhibit_g = relationship("ExhibitG", back_populates="transcription_runs")


class CellDisagreement(Base):
    """Tracks disagreements between OCR reads."""
    __tablename__ = "cell_disagreement"

    id = Column(Integer, primary_key=True, index=True)
    exhibit_g_id = Column(Integer, ForeignKey("exhibit_g.id"), nullable=False)
    row_number = Column(Integer, nullable=False)
    column_name = Column(String(50), nullable=False)

    # Values from different engines
    values = Column(JSON, nullable=False)  # {"paddleocr": "7:30A", "easyocr": "7:30a", "claude": "7:30A"}

    resolved_value = Column(Text, nullable=True)
    is_resolved = Column(Boolean, default=False)

    exhibit_g = relationship("ExhibitG", back_populates="disagreements")
