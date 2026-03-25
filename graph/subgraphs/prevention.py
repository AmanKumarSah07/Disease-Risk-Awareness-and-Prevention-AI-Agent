from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.state import AgentState

load_dotenv()

def _get_query_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    return str(messages[-1]) if messages else ""

def prevention_plan_node(state: AgentState) -> AgentState:
    profile = state.get("user_profile") or {}
    symptoms: List[str] = state.get("symptoms") or []
    risk_scores = state.get("risk_scores") or {}
    query = _get_query_text(state)

    top = list(risk_scores.items())[:3]
    top_text = ", ".join([f"{d} ({s:.2f})" for d, s in top]) if top else "none"

    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # Deterministic fallback
    if not groq_key:
        text = (
            "Here’s a practical prevention plan (general guidance, not medical advice):\n\n"
            "1) Basics\n"
            "- Hydration, regular sleep, and balanced meals.\n"
            "- If you have fever, rest and monitor symptoms.\n\n"
            "2) Lifestyle\n"
            "- Aim for 150 minutes/week of moderate activity (walk/cycle), if safe for you.\n"
            "- Reduce ultra-processed foods; prefer vegetables, fruit, whole grains, and protein.\n\n"
            "3) Risk-specific pointers\n"
            f"- Top risk signals (if any): {top_text}\n"
            "- If mosquito-borne diseases are a concern in your area, use repellant + long sleeves + remove stagnant water.\n\n"
            "4) When to see a clinician\n"
            "- If symptoms are severe, persistent, or worsening, seek medical care.\n\n"
            "If you share your age, location, and any conditions, I can make this more personalized."
        )
        state["prevention_plan"] = text
        state["final_response"] = text
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": text})
        state["messages"] = msgs
        return state

    # LLM plan
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

        prompt = (
            "You are a prevention-focused health assistant. Do NOT diagnose.\n"
            "Write a personalized, practical prevention plan based on the user's question, profile, symptoms, and top risks.\n"
            "Include:\n"
            "- daily actions (diet, exercise, sleep)\n"
            "- region-specific prevention if location suggests it\n"
            "- red flags to seek medical care\n\n"
            f"User question: {query}\n"
            f"Profile: {profile}\n"
            f"Symptoms: {symptoms}\n"
            f"Top risks: {top_text}\n"
        )

        text = llm.invoke(prompt).content
        state["prevention_plan"] = text
        state["final_response"] = text
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": text})
        state["messages"] = msgs
        return state
    except Exception:
        text = (
            "I couldn’t generate a prevention plan right now. "
            "Please try again, or tell me your age, location, and any medical conditions for more tailored guidance."
        )
        state["prevention_plan"] = text
        state["final_response"] = text
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": text})
        state["messages"] = msgs
        return state

def build_prevention_subgraph():
    g = StateGraph(AgentState)
    g.add_node("plan", prevention_plan_node)
    g.set_entry_point("plan")
    g.add_edge("plan", END)
    return g.compile()

prevention_subgraph = build_prevention_subgraph()