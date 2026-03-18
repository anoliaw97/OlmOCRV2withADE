from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    report_name = Column(String(255), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="uploaded")


class ExtractedTable(Base):
    __tablename__ = "extracted_tables"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    page_number = Column(Integer, nullable=False)
    table_id = Column(String(50), nullable=False)
    extraction_type = Column(String(100), nullable=False)
    table_title = Column(String(500), nullable=True)
    columns_json = Column(JSON, nullable=False)
    rows_json = Column(JSON, nullable=False)
    units_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    report = relationship("Report")


class RagChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, nullable=False)
    table_ref_id = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=False)
    keyword_score_hint = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, nullable=True)
    stage = Column(String(80), nullable=False)
    level = Column(String(20), default="info")
    message = Column(Text, nullable=False)
    payload_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
