from __future__ import annotations

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from graph.state import AgentState

load_dotenv()

class IntentClassification(BaseModel):
    intent: Literal["risk", "info", "prevention"] = Field(
        ...,
        description="Classify the user's intent as risk, info, or prevention."
    )

def _rule_based_intent(text: str) -> str:
    t = (text or "").lower()

    risk_triggers = ["symptom", "symptoms", "am i at risk", "risk", "i feel", "pain", "fever", "cough", "headache"]
    info_triggers = ["what is", "causes", "spread", "transmit", "contagious", "incubation", "how does", "symptoms of"]
    prevention_triggers = ["prevent", "avoid", "reduce risk", "diet", "exercise", "vaccine", "vaccination", "screening"]

    if any(k in t for k in prevention_triggers):
        return "prevention"
    if any(k in t for k in info_triggers):
        return "info"
    if any(k in t for k in risk_triggers):
        return "risk"
    # default
    return "info"

def router_node(state: AgentState) -> AgentState:
    """
    Intent classification node.
    - Uses Groq LLM when available
    - Falls back to rule-based classification if no API key or LLM failure
    """
    messages = state.get("messages") or []
    last_user_text: str = ""
    if messages:
        # messages may include dicts or LangChain message objects; str() is safe
        last_user_text = str(messages[-1])

    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # Fallback if key missing
    if not groq_key:
        state["intent"] = _rule_based_intent(last_user_text)
        return state

    # LLM-based router
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        structured = llm.with_structured_output(IntentClassification)

        prompt = (
            "Classify the user's last message into exactly one intent:\n"
            "- risk: user describes symptoms or asks if they are at risk\n"
            "- info: user asks about a disease/topic, causes, spread, facts\n"
            "- prevention: user asks how to prevent disease or improve health\n\n"
            f"User message: {last_user_text}"
        )

        result: IntentClassification = structured.invoke(prompt)
        state["intent"] = result.intent
        return state

    except Exception:
        # Always fail safely to rule-based routing
        state["intent"] = _rule_based_intent(last_user_text)
        return state