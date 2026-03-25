from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "./checkpoints.db")

parent = os.path.dirname(CHECKPOINT_DB_PATH)
if parent:
    os.makedirs(parent, exist_ok=True)

conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)