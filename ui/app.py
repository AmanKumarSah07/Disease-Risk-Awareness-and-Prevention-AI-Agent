from __future__ import annotations

import uuid

import requests
import streamlit as st

# Safe secrets access (Streamlit raises if no secrets.toml exists)
try:
    API_BASE_DEFAULT = st.secrets.get("API_BASE")  # type: ignore[attr-defined]
except Exception:
    API_BASE_DEFAULT = None

API_BASE_DEFAULT = API_BASE_DEFAULT or "http://127.0.0.1:8000"

st.set_page_config(page_title="Disease Agent", layout="centered")
st.title("Disease Agent")

# Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
if "chat" not in st.session_state:
    st.session_state.chat = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "testuser"

with st.sidebar:
    st.subheader("Connection")
    api_base = st.text_input("API base", API_BASE_DEFAULT)
    st.caption(f"Thread ID: {st.session_state.thread_id}")

    st.subheader("User Profile")
    user_id = st.text_input("user_id", st.session_state.user_id)
    st.session_state.user_id = user_id

    col1, col2 = st.columns(2)
    if col1.button("Load profile"):
        r = requests.get(f"{api_base}/profile/{user_id}")
        if r.status_code == 200:
            st.session_state.profile = r.json()
            st.success("Loaded.")
        else:
            st.session_state.profile = {"user_id": user_id}
            st.warning("No profile found; using empty.")

    if col2.button("Save profile"):
        profile = st.session_state.get("profile") or {"user_id": user_id}
        profile["user_id"] = user_id
        r = requests.put(f"{api_base}/profile/{user_id}", json=profile)
        if r.status_code == 200:
            st.success("Saved.")
        else:
            st.error(r.text)

    profile = st.session_state.get("profile") or {"user_id": user_id}
    profile["user_id"] = user_id

    profile["age"] = st.number_input(
        "age",
        value=int(profile.get("age") or 0),
        min_value=0,
        step=1,
    )
    profile["sex"] = st.text_input("sex", value=profile.get("sex") or "")
    profile["location"] = st.text_input("location", value=profile.get("location") or "")

    conditions_str = st.text_input(
        "conditions (comma-separated)",
        value=",".join(profile.get("conditions") or []),
    )
    meds_str = st.text_input(
        "medications (comma-separated)",
        value=",".join(profile.get("medications") or []),
    )

    profile["conditions"] = [x.strip() for x in conditions_str.split(",") if x.strip()]
    profile["medications"] = [x.strip() for x in meds_str.split(",") if x.strip()]
    profile["lifestyle"] = profile.get("lifestyle") or {}

    st.session_state.profile = profile

st.divider()

# Render chat history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Ask about symptoms, disease info, or prevention...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "thread_id": st.session_state.thread_id,
        "message": prompt,
        "intent": None,
        "user_profile": st.session_state.get("profile"),
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = requests.post(f"{api_base}/chat", json=payload)
            if r.status_code != 200:
                st.error(r.text)
            else:
                data = r.json()
                answer = data.get("final_response", "")
                st.markdown(answer)
                st.session_state.chat.append({"role": "assistant", "content": answer})