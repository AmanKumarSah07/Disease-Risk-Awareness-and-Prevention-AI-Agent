# Disease Agent (LangGraph + FastAPI + Streamlit)

A small “disease assistant” demo that routes user queries into:
- **info** (general disease info)
- **risk** (risk scoring / triage-style guidance)
- **prevention** (prevention plan)

It includes:
- LangGraph state machine + SQLite checkpointing
- FastAPI API (`/chat`, `/profile/{user_id}`)
- Streamlit UI talking to the API

## Requirements
- Python 3.12 recommended

## Setup

```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file (copy from `.env.example`):

```bash
copy .env.example .env
```

## Run (2 terminals)

### Terminal 1: FastAPI
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open API docs:
- http://127.0.0.1:8000/docs

### Terminal 2: Streamlit UI
```bash
streamlit run ui/app.py
```

## API

### POST /chat
Body:
```json
{
  "thread_id": "t1",
  "message": "What is dengue fever?",
  "intent": "info",
  "user_profile": { "user_id": "testuser" }
}
```

### GET /profile/{user_id}
Fetch saved profile.

### PUT /profile/{user_id}
Create/update profile.

## Notes
- Profiles are stored in `SQLITE_DB_PATH` (default `./agent_memory.db`).
- Conversation checkpoints are stored in `CHECKPOINT_DB_PATH` (default `./checkpoints.db`).