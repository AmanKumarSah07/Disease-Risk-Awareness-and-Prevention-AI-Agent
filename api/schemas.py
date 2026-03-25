from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field

Intent = Optional[Literal["risk", "info", "prevention"]]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str

class UserProfileIn(BaseModel):
    user_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    location: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    lifestyle: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Conversation id for checkpointing")
    message: str
    intent: Intent = None
    user_profile: Optional[UserProfileIn] = None

class ChatResponse(BaseModel):
    thread_id: str
    intent: Intent
    final_response: str
    risk_scores: Dict[str, float] = Field(default_factory=dict)
    prevention_plan: Optional[str] = None