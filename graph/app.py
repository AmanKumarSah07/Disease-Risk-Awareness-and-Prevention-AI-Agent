from __future__ import annotations

from typing import Optional

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.router import router_node
from graph.guardrail import guardrail_node
from graph.subgraphs.risk import risk_subgraph
from graph.subgraphs.info import info_subgraph
from graph.subgraphs.prevention import prevention_subgraph
from memory.checkpointer import checkpointer
from memory.profile_store import get_profile

def load_profile_node(state: AgentState) -> AgentState:
    """
    Load profile from SQLite if state.user_profile contains user_id OR if messages contain it later.
    Expect caller to set: state['user_profile']={'user_id': '...'} at minimum.
    """
    prof = state.get("user_profile") or {}
    user_id = prof.get("user_id")
    if not user_id:
        return state

    db_prof = get_profile(user_id)
    if db_prof:
        state["user_profile"] = db_prof  # hydrate full profile
    else:
        # keep provided stub
        state["user_profile"] = prof
    return state

def route_to_subgraph(state: AgentState) -> str:
    intent = state.get("intent") or "info"
    if intent == "risk":
        return "risk"
    if intent == "prevention":
        return "prevention"
    return "info"

def run_risk(state: AgentState) -> AgentState:
    return risk_subgraph.invoke(state)

def run_info(state: AgentState) -> AgentState:
    return info_subgraph.invoke(state)

def run_prevention(state: AgentState) -> AgentState:
    return prevention_subgraph.invoke(state)

def build_app():
    g = StateGraph(AgentState)

    g.add_node("load_profile", load_profile_node)
    g.add_node("router", router_node)

    g.add_node("risk", run_risk)
    g.add_node("info", run_info)
    g.add_node("prevention", run_prevention)

    g.add_node("guardrail", guardrail_node)

    g.set_entry_point("load_profile")
    g.add_edge("load_profile", "router")

    g.add_conditional_edges(
        "router",
        route_to_subgraph,
        {"risk": "risk", "info": "info", "prevention": "prevention"},
    )

    g.add_edge("risk", "guardrail")
    g.add_edge("info", "guardrail")
    g.add_edge("prevention", "guardrail")

    g.add_edge("guardrail", END)

    return g.compile(checkpointer=checkpointer)

app_graph = build_app()