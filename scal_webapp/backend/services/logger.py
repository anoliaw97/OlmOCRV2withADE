from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import ProcessingLog


def log_event(db: Session, stage: str, message: str, report_id: int | None = None, level: str = "info", payload: dict | None = None):
    row = ProcessingLog(
        report_id=report_id,
        stage=stage,
        level=level,
        message=message,
        payload_json=payload or {},
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
