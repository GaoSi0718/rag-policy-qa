"""Single-user SQLite storage. Diagnostic traces omit prompts and tool values."""
import json
import sqlite3
import time
from contextlib import contextmanager

from .config import DB

@contextmanager
def connection():
    con = sqlite3.connect(DB, timeout=0.25)
    try:
        with con:
            yield con
    finally:
        con.close()

def initialize():
    DB.parent.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS turns (
                run_id TEXT PRIMARY KEY, session TEXT, ts REAL, question TEXT, answer TEXT
            );
            CREATE INDEX IF NOT EXISTS turns_session ON turns(session, ts);
            CREATE TABLE IF NOT EXISTS traces (
                run_id TEXT, seq INTEGER, event TEXT, PRIMARY KEY (run_id, seq)
            );
        """)

def history(session):
    with connection() as con:
        rows = con.execute(
            "SELECT question, answer FROM turns WHERE session=? ORDER BY ts DESC, rowid DESC LIMIT 20",
            (session,),
        ).fetchall()
    return list(reversed(rows))

def complete(run_id, session, question, result, trace):
    with connection() as con:
        con.executemany(
            "INSERT INTO traces VALUES (?, ?, ?)",
            [(run_id, seq, json.dumps(item, ensure_ascii=False)) for seq, item in enumerate(trace)],
        )
        if result and result["status"] in {"answered", "refused", "clarify"}:
            con.execute("INSERT INTO turns VALUES (?, ?, ?, ?, ?)",
                        (run_id, session, time.time(), question, result["answer"]))

def read_trace(run_id):
    with connection() as con:
        rows = con.execute("SELECT event FROM traces WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
    return [json.loads(row[0]) for row in rows]
