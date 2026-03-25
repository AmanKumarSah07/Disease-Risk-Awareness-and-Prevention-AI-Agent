from __future__ import annotations

import json
import os
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "./agent_memory.db")

# Ensure parent dir exists (in case a path like ./data/db/agent_memory.db is used)
parent = os.path.dirname(DB_PATH)
if parent:
    os.makedirs(parent, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},  # required for FastAPI multithreading
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

class UserProfileDB(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True, index=True)

    age = Column(Integer, nullable=True)
    sex = Column(String, nullable=True)
    location = Column(String, nullable=True)

    # Stored as JSON strings
    conditions = Column(Text, nullable=False, default="[]")
    medications = Column(Text, nullable=False, default="[]")
    lifestyle = Column(Text, nullable=False, default="{}")

def _json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value) if value else fallback
    except Exception:
        return fallback

def _json_dumps(value: Any, fallback: str) -> str:
    try:
        return json.dumps(value if value is not None else json.loads(fallback))
    except Exception:
        return fallback

# Initialize DB tables on import
Base.metadata.create_all(bind=engine)

def get_profile(user_id: str) -> Optional[dict]:
    with SessionLocal() as db:
        row = db.query(UserProfileDB).filter(UserProfileDB.user_id == user_id).first()
        if not row:
            return None
        return {
            "user_id": row.user_id,
            "age": row.age,
            "sex": row.sex,
            "location": row.location,
            "conditions": _json_loads(row.conditions, []),
            "medications": _json_loads(row.medications, []),
            "lifestyle": _json_loads(row.lifestyle, {}),
        }

def create_or_update_profile(user_id: str, profile_data: dict) -> None:
    """
    Create a profile if it doesn't exist, then update provided fields.
    This is safe to call repeatedly.
    """
    with SessionLocal() as db:
        row = db.query(UserProfileDB).filter(UserProfileDB.user_id == user_id).first()
        if not row:
            row = UserProfileDB(user_id=user_id)
            db.add(row)

        if "age" in profile_data:
            row.age = profile_data["age"]
        if "sex" in profile_data:
            row.sex = profile_data["sex"]
        if "location" in profile_data:
            row.location = profile_data["location"]
        if "conditions" in profile_data:
            row.conditions = _json_dumps(profile_data["conditions"], "[]")
        if "medications" in profile_data:
            row.medications = _json_dumps(profile_data["medications"], "[]")
        if "lifestyle" in profile_data:
            row.lifestyle = _json_dumps(profile_data["lifestyle"], "{}")

        db.commit()

def update_field(user_id: str, field: str, value: Any) -> None:
    """
    Update a single field. Creates the user profile row if missing.
    """
    allowed = {"age", "sex", "location", "conditions", "medications", "lifestyle"}
    if field not in allowed:
        raise ValueError(f"Invalid field '{field}'. Allowed: {sorted(allowed)}")

    with SessionLocal() as db:
        row = db.query(UserProfileDB).filter(UserProfileDB.user_id == user_id).first()
        if not row:
            row = UserProfileDB(user_id=user_id)
            db.add(row)

        if field in {"conditions", "medications"}:
            setattr(row, field, _json_dumps(value, "[]"))
        elif field == "lifestyle":
            setattr(row, field, _json_dumps(value, "{}"))
        else:
            setattr(row, field, value)

        db.commit()