from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ChatRequest, ChatResponse
from graph.app import app_graph

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = {
        "messages": [{"role": "user", "content": req.message}],
        "intent": req.intent,
        "user_profile": req.user_profile.model_dump() if req.user_profile else None,
        "symptoms": [],
        "risk_scores": {},
        "retrieved_docs": [],
        "prevention_plan": None,
        "safety_flag": False,
        "final_response": None,
    }

    try:
        out = app_graph.invoke(
            state,
            config={"configurable": {"thread_id": req.thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    final_text = out.get("final_response") or ""
    return ChatResponse(
        thread_id=req.thread_id,
        intent=out.get("intent"),
        final_response=final_text,
        risk_scores=out.get("risk_scores") or {},
        prevention_plan=out.get("prevention_plan"),
    )