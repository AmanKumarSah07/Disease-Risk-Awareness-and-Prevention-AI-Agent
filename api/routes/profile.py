from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import UserProfileIn
from memory.profile_store import create_or_update_profile, get_profile

router = APIRouter()

@router.get("/profile/{user_id}")
def read_profile(user_id: str):
    prof = get_profile(user_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found")
    return prof

@router.put("/profile/{user_id}")
def write_profile(user_id: str, profile: UserProfileIn):
    if profile.user_id != user_id:
        raise HTTPException(status_code=400, detail="user_id mismatch")
    create_or_update_profile(user_id, profile.model_dump())
    return {"ok": True}