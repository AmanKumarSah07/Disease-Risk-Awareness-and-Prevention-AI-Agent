from __future__ import annotations

import os
import re
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from graph.state import AgentState, UserProfile
from tools.risk_scorer import calculate_risk_score

load_dotenv()

# -------------------------
# Helpers
# -------------------------

REQUIRED_PROFILE_FIELDS = ["age", "location", "conditions"]

def _get_profile(state: AgentState) -> UserProfile:
    prof = state.get("user_profile") or {}
    # ensure required keys exist
    prof.setdefault("user_id", prof.get("user_id") or "unknown")
    prof.setdefault("age", prof.get("age"))
    prof.setdefault("sex", prof.get("sex"))
    prof.setdefault("location", prof.get("location"))
    prof.setdefault("conditions", prof.get("conditions") or [])
    prof.setdefault("medications", prof.get("medications") or [])
    prof.setdefault("lifestyle", prof.get("lifestyle") or {})
    return prof  # type: ignore[return-value]

def _missing_fields(profile: UserProfile) -> list[str]:
    missing = []
    if profile.get("age") in (None, "", 0):
        missing.append("age")
    if not profile.get("location"):
        missing.append("location")
    # conditions can be empty list; that's OK, but we still want to ask explicitly at least once
    if profile.get("conditions") is None:
        missing.append("conditions")
    return missing

def _append_ai_message(state: AgentState, text: str) -> AgentState:
    msgs = state.get("messages") or []
    msgs.append({"role": "assistant", "content": text})
    state["messages"] = msgs
    return state

# -------------------------
# Nodes
# -------------------------

def profile_collection_node(state: AgentState) -> AgentState:
    profile = _get_profile(state)
    state["user_profile"] = profile

    missing = _missing_fields(profile)
    if not missing:
        return state

    # Ask only for the first missing field to keep it conversational
    field = missing[0]
    if field == "age":
        q = "To help assess risk more accurately, what is your age?"
    elif field == "location":
        q = "What city/state (or country) are you currently in? This helps with region-specific risks."
    else:  # conditions
        q = "Do you have any pre-existing conditions (e.g., diabetes, hypertension, asthma)? If none, you can say 'none'."

    state = _append_ai_message(state, q)
    # We stop here; caller should send user response next turn.
    state["final_response"] = q
    return state

def is_profile_complete(state: AgentState) -> str:
    profile = _get_profile(state)
    missing = _missing_fields(profile)
    return "complete" if not missing else "incomplete"

class SymptomExtraction(BaseModel):
    symptoms: List[str] = Field(default_factory=list)

def _heuristic_symptom_extract(text: str) -> list[str]:
    # Very basic fallback: extract common symptom phrases
    candidates = [
        "fever", "cough", "sore throat", "headache", "body aches", "fatigue",
        "shortness of breath", "difficulty breathing", "chest pain",
        "nausea", "vomiting", "diarrhea", "abdominal pain", "rash",
        "chills", "night sweats", "weight loss", "dizziness",
    ]
    t = (text or "").lower()
    out = []
    for c in candidates:
        if c in t:
            out.append(c)
    return sorted(set(out))

def symptom_extraction_node(state: AgentState) -> AgentState:
    messages = state.get("messages") or []
    convo_text = "\n".join([str(m) for m in messages[-12:]])  # last N messages

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        state["symptoms"] = _heuristic_symptom_extract(convo_text)
        return state

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        structured = llm.with_structured_output(SymptomExtraction)

        prompt = (
            "Extract the user's symptoms from the conversation. "
            "Return a JSON object with a 'symptoms' list of concise symptom strings. "
            "Do not include diagnoses. If none, return an empty list.\n\n"
            f"Conversation:\n{convo_text}"
        )
        res: SymptomExtraction = structured.invoke(prompt)
        # normalize
        symptoms = [" ".join(s.strip().lower().split()) for s in res.symptoms if s and s.strip()]
        state["symptoms"] = sorted(set(symptoms))
        return state
    except Exception:
        state["symptoms"] = _heuristic_symptom_extract(convo_text)
        return state

def risk_scoring_node(state: AgentState) -> AgentState:
    profile = _get_profile(state)
    symptoms = state.get("symptoms") or []
    scores = calculate_risk_score.invoke({"symptoms": symptoms, "profile": profile})
    state["risk_scores"] = scores or {}
    return state

def response_generation_node(state: AgentState) -> AgentState:
    profile = _get_profile(state)
    symptoms = state.get("symptoms") or []
    scores = state.get("risk_scores") or {}

    # pick top 3
    top = list(scores.items())[:3]

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        # deterministic fallback response
        if not top:
            text = (
                "Based on what you shared, I can’t identify a strong pattern from common symptom groupings. "
                "If your symptoms are severe or getting worse, please see a clinician.\n\n"
                f"Symptoms captured: {', '.join(symptoms) if symptoms else 'none'}.\n"
                "If you want, share your age, location, and any conditions to refine the risk check."
            )
        else:
            bullets = "\n".join([f"- {d}: {s:.2f}" for d, s in top])
            text = (
                "Here’s a risk-aware summary (not a diagnosis):\n\n"
                f"{bullets}\n\n"
                "These scores reflect symptom overlap + basic modifiers (age/location/conditions). "
                "For a definitive assessment, please consult a doctor—especially if symptoms worsen."
            )
        state["final_response"] = text
        state = _append_ai_message(state, text)
        return state

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

        top_text = ", ".join([f"{d} ({s:.2f})" for d, s in top]) if top else "none"

        prompt = (
            "You are a disease risk awareness assistant. Do NOT diagnose.\n"
            "Given symptoms, user profile, and risk scores, write a calm, non-alarmist response.\n"
            "Must include:\n"
            "1) Top 2-3 risk factors/diseases (if any)\n"
            "2) What the scores mean in plain language\n"
            "3) Recommend consulting a doctor for definitive assessment\n\n"
            f"User profile: age={profile.get('age')}, location={profile.get('location')}, conditions={profile.get('conditions')}\n"
            f"Symptoms: {symptoms}\n"
            f"Top risks: {top_text}\n"
            f"All risk scores: {scores}\n"
        )

        text = llm.invoke(prompt).content
        state["final_response"] = text
        state = _append_ai_message(state, text)
        return state
    except Exception:
        # fallback template
        bullets = "\n".join([f"- {d}: {s:.2f}" for d, s in top]) if top else "- No strong matches found."
        text = (
            "Here’s a risk-aware summary (not a diagnosis):\n\n"
            f"{bullets}\n\n"
            "If your symptoms are severe, sudden, or worsening, please seek medical care. "
            "A clinician can evaluate your history, exam findings, and tests."
        )
        state["final_response"] = text
        state = _append_ai_message(state, text)
        return state

# -------------------------
# Graph builder
# -------------------------

def build_risk_subgraph():
    g = StateGraph(AgentState)

    g.add_node("collect_profile", profile_collection_node)
    g.add_node("extract_symptoms", symptom_extraction_node)
    g.add_node("score_risk", risk_scoring_node)
    g.add_node("generate_response", response_generation_node)

    # entry
    g.set_entry_point("collect_profile")

    # loop until profile is complete
    g.add_conditional_edges(
        "collect_profile",
        is_profile_complete,
        {
            "incomplete": END,        # stop this run; user must respond next turn
            "complete": "extract_symptoms",
        },
    )

    g.add_edge("extract_symptoms", "score_risk")
    g.add_edge("score_risk", "generate_response")
    g.add_edge("generate_response", END)

    return g.compile()

risk_subgraph = build_risk_subgraph()

if __name__ == "__main__":
    # Simple smoke test (no LLM required)
    state: AgentState = {
        "messages": [{"role": "user", "content": "I have fever, chills and headache."}],
        "intent": "risk",
        "user_profile": {
            "user_id": "demo",
            "age": 35,
            "sex": "M",
            "location": "Patna, Bihar",
            "conditions": ["smoking"],
            "medications": [],
            "lifestyle": {"smoking": True, "exercise": "low", "diet": "mixed"},
        },
        "symptoms": [],
        "risk_scores": {},
        "retrieved_docs": [],
        "prevention_plan": None,
        "safety_flag": False,
        "final_response": None,
    }
    out = risk_subgraph.invoke(state)
    print(out.get("risk_scores"))
    print(out.get("final_response"))