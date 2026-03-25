from __future__ import annotations

import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "./checkpoints.db")

# Sync saver works in normal (non-async) scripts.
from langgraph.checkpoint.sqlite import SqliteSaver

_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)