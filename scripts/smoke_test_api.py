from __future__ import annotations

import sys
import requests

API = "http://127.0.0.1:8000"
USER_ID = "smokeuser"
THREAD_ID = "smokethread-1"

def main() -> int:
    # PUT profile
    prof = {
        "user_id": USER_ID,
        "age": 25,
        "sex": "M",
        "location": "Delhi",
        "conditions": [],
        "medications": [],
        "lifestyle": {},
    }
    r = requests.put(f"{API}/profile/{USER_ID}", json=prof)
    print("PUT /profile:", r.status_code, r.text)
    if r.status_code != 200:
        return 1

    # GET profile
    r = requests.get(f"{API}/profile/{USER_ID}")
    print("GET /profile:", r.status_code, r.text[:200])
    if r.status_code != 200:
        return 1

    # POST chat
    payload = {
        "thread_id": THREAD_ID,
        "message": "What is dengue fever?",
        "intent": "info",
        "user_profile": {"user_id": USER_ID},
    }
    r = requests.post(f"{API}/chat", json=payload)
    print("POST /chat:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return 1

    data = r.json()
    print("intent:", data.get("intent"))
    print("final_response (first 200 chars):", (data.get("final_response") or "")[:200])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())