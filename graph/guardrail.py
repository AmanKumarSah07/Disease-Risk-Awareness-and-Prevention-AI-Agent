from __future__ import annotations

from graph.state import AgentState

EMERGENCY_SYMPTOMS = {
    "chest pain",
    "difficulty breathing",
    "stroke",
    "seizure",
    "severe bleeding",
    "unconscious",
    "suicidal",
}

EMERGENCY_KEYWORDS = {
    "emergency",
    "call an ambulance",
    "call ambulance",
    "cant breathe",
    "can't breathe",
    "heart attack",
    "stroke",
    "passed out",
    "fainted",
    "unconscious",
    "suicidal",
    "kill myself",
}

def guardrail_node(state: AgentState) -> AgentState:
    """
    Deterministic safety node (NO LLM).
    Flags if:
      - any risk score > 0.80
      - emergency symptoms present
      - user mentions emergency explicitly
    If flagged, overwrites final_response with urgent guidance.
    """
    symptoms = [str(s).lower().strip() for s in (state.get("symptoms") or [])]
    risk_scores = state.get("risk_scores") or {}
    messages = state.get("messages") or []

    high_risk = any((score is not None and float(score) > 0.80) for score in risk_scores.values())
    emergency_symptom_found = any(es in symptoms for es in EMERGENCY_SYMPTOMS)

    last_text = " ".join([str(m) for m in messages[-3:]]).lower()
    emergency_mentioned = any(k in last_text for k in EMERGENCY_KEYWORDS)

    safety_flag = bool(high_risk or emergency_symptom_found or emergency_mentioned)
    state["safety_flag"] = safety_flag

    if safety_flag:
        state["final_response"] = (
            "I’m really sorry you’re dealing with this. I can’t provide a medical diagnosis, "
            "but your symptoms/risk indicators could be serious.\n\n"
            "**Please seek urgent medical care now**:\n"
            "- If you think this is an emergency, call your local emergency number immediately.\n"
            "- Otherwise, go to the nearest emergency department / urgent care or contact a doctor right away.\n\n"
            "If you are in India, you can also contact the AIIMS helpline: **1800-11-7744**.\n"
            "For general health guidance, you can refer to WHO resources: **who.int**.\n\n"
            "If you tell me your location and what you’re experiencing right now, I can help you find the most appropriate next step."
        )

    return state