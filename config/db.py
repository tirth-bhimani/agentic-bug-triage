import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "trace.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT,
            repo TEXT,
            issue INTEGER,
            stage TEXT,
            input TEXT,
            output TEXT,
            tokens INTEGER,
            latency_ms INTEGER,
            ts REAL
        )
    """)
    conn.commit()
    conn.close()
    print(f"DB initialized at {DB_PATH}")


def trace(run_id, repo, issue, stage, input_data=None, output_data=None, tokens=0, latency_ms=0):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO runs (run_id, repo, issue, stage, input, output, tokens, latency_ms, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            repo,
            issue,
            stage,
            json.dumps(input_data) if input_data is not None else None,
            json.dumps(output_data) if output_data is not None else None,
            tokens,
            latency_ms,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def get_run_trace(run_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM runs WHERE run_id = ? ORDER BY ts ASC", (run_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()