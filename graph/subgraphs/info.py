from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from graph.state import AgentState
from tools.web_search import medical_web_search

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

def _get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _get_query_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    return str(messages[-1]) if messages else ""

def retrieve_node(state: AgentState) -> AgentState:
    query = _get_query_text(state).strip()
    if not query:
        state["retrieved_docs"] = []
        return state

    # If vector store doesn't exist yet, skip retrieval
    if not os.path.exists(CHROMA_PERSIST_DIR):
        state["retrieved_docs"] = []
        return state

    vectordb = Chroma(
        collection_name="medical_knowledge",
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=_get_embeddings(),
    )
    docs = vectordb.similarity_search(query, k=4)
    state["retrieved_docs"] = [d.page_content for d in docs]
    return state

def answer_node(state: AgentState) -> AgentState:
    query = _get_query_text(state)
    retrieved: List[str] = state.get("retrieved_docs") or []

    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    # If no docs, do web fallback
    web = ""
    if not retrieved:
        try:
            web = medical_web_search.invoke({"query": query})
        except Exception:
            web = ""

    # Fallback (no LLM)
    if not groq_key:
        if retrieved:
            context = "\n\n".join(retrieved[:2])
            text = (
                "Here’s what I found in the uploaded knowledge base (not medical advice):\n\n"
                f"{context}\n\n"
                "If you want, tell me which disease/topic you mean and I can summarize it more clearly."
            )
        else:
            text = (
                "I don’t have local documents ingested yet. Here are some web search results you can open:\n\n"
                f"{web}\n\n"
                "If you add PDFs into `data/raw/` and run `py -m data.ingest`, I can answer from your documents too."
            )
        state["final_response"] = text
        (state.get("messages") or []).append({"role": "assistant", "content": text})
        state["messages"] = state.get("messages") or []
        return state

    # LLM answer
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

        context_block = "\n\n".join(retrieved[:4])
        prompt = (
            "You are a health information assistant. Do NOT diagnose.\n"
            "Answer the user's question clearly and concisely.\n"
            "If context is provided, use it; otherwise use the web results.\n"
            "Encourage consulting a medical professional for personal medical decisions.\n\n"
            f"Question: {query}\n\n"
            f"Context from documents:\n{context_block}\n\n"
            f"Web results (if any):\n{web}\n"
        )

        text = llm.invoke(prompt).content
        state["final_response"] = text
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": text})
        state["messages"] = msgs
        return state
    except Exception:
        text = (
            "I couldn’t generate an LLM-based answer right now. "
            "Try again, or ingest PDFs and I’ll answer from the local knowledge base."
        )
        state["final_response"] = text
        msgs = state.get("messages") or []
        msgs.append({"role": "assistant", "content": text})
        state["messages"] = msgs
        return state

def build_info_subgraph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

    return g.compile()

info_subgraph = build_info_subgraph()