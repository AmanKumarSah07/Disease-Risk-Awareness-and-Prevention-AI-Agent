from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages

class UserProfile(TypedDict):
    user_id: str
    age: Optional[int]
    sex: Optional[str]
    location: Optional[str]
    conditions: list[str]        # pre-existing conditions
    medications: list[str]
    lifestyle: dict              # {"smoking": bool, "exercise": str, "diet": str}

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: Optional[str]                      # "risk" | "info" | "prevention"
    user_profile: Optional[UserProfile]
    symptoms: list[str]                        # symptoms reported this session
    risk_scores: dict[str, float]              # {"diabetes": 0.72, "hypertension": 0.45}
    retrieved_docs: list[str]                  # RAG results
    prevention_plan: Optional[str]
    safety_flag: bool                          # True = escalate to doctor
    final_response: Optional[str]