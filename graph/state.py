from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Literal

Intent = Optional[Literal["risk", "info", "prevention"]]

class UserProfile(TypedDict, total=False):
    user_id: str
    age: Optional[int]
    sex: Optional[str]
    location: Optional[str]
    conditions: List[str]
    medications: List[str]
    lifestyle: Dict[str, Any]

class AgentState(TypedDict, total=False):
    # Conversation
    messages: List[Any]  # can be dicts or LangChain messages
    intent: Intent

    # Memory/profile
    user_profile: Optional[UserProfile]

    # Risk pipeline
    symptoms: List[str]
    risk_scores: Dict[str, float]

    # Info pipeline (RAG)
    retrieved_docs: List[str]

    # Prevention pipeline
    prevention_plan: Optional[str]

    # Safety / final
    safety_flag: bool
    final_response: Optional[str]