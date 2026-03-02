"""
db.py — Database, SM-2 algorithm, streak tracking, and progress reset.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def get_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            problem_id  TEXT    PRIMARY KEY,
            interval    REAL    DEFAULT 1,
            easiness    REAL    DEFAULT 2.5,
            reps        INTEGER DEFAULT 0,
            next_review TEXT    DEFAULT (datetime('now')),
            last_output TEXT    DEFAULT ''
        )
    """)
    # one row per calendar day a review was submitted
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            day TEXT PRIMARY KEY
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mistakes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT NOT NULL,
            summary    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cols = {row[1] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()}
    if "last_rating" not in cols:
        conn.execute("ALTER TABLE reviews ADD COLUMN last_rating INTEGER")
    conn.commit()
    return conn


def get_row(conn: sqlite3.Connection, pid: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM reviews WHERE problem_id=?", (pid,)
    ).fetchone()


def sm2_update(conn: sqlite3.Connection, pid: str, rating: int) -> None:
    """SM-2 algorithm. rating: 1=Again 2=Hard 3=Good 4=Easy"""
    quality = {1: 0, 2: 3, 3: 4, 4: 5}[rating]
    row = get_row(conn, pid)
    iv, ef, reps = (row["interval"], row["easiness"], row["reps"]) if row else (1.0, 2.5, 0)

    if quality < 3:
        iv, reps = 1.0, 0
    else:
        iv   = 1 if reps == 0 else (6 if reps == 1 else round(iv * ef, 1))
        reps += 1
        ef   = max(1.3, ef + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

    next_rev = (datetime.now() + timedelta(days=iv)).isoformat()
    conn.execute("""
        INSERT INTO reviews (problem_id, interval, easiness, reps, next_review, last_rating)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(problem_id) DO UPDATE SET
            interval=excluded.interval,
            easiness=excluded.easiness,
            reps=excluded.reps,
            next_review=excluded.next_review,
            last_rating=excluded.last_rating
    """, (pid, iv, ef, reps, next_rev, rating))
    conn.execute(
        "INSERT OR IGNORE INTO activity (day) VALUES (?)",
        (datetime.now().date().isoformat(),),
    )
    conn.commit()


def get_streak(conn: sqlite3.Connection) -> int:
    """Return current consecutive-day streak (today counts if active today)."""
    rows = conn.execute(
        "SELECT day FROM activity ORDER BY day DESC"
    ).fetchall()
    if not rows:
        return 0
    days = [datetime.fromisoformat(r["day"]).date() for r in rows]
    today = datetime.now().date()
    streak = 0
    cursor = today
    for day in days:
        if day == cursor:
            streak += 1
            cursor -= timedelta(days=1)
        elif day == cursor + timedelta(days=1):
            streak += 1
            cursor = day - timedelta(days=1)
        else:
            break
    return streak


def reset_progress(conn: sqlite3.Connection, pid: str) -> None:
    conn.execute("DELETE FROM reviews WHERE problem_id=?", (pid,))
    conn.execute("DELETE FROM mistakes WHERE problem_id=?", (pid,))
    conn.commit()


def log_mistake(conn: sqlite3.Connection, pid: str, summary: str) -> None:
    text = summary.strip()
    if not text:
        return
    conn.execute(
        "INSERT INTO mistakes (problem_id, summary) VALUES (?, ?)",
        (pid, text),
    )
    conn.commit()


def recent_mistakes(conn: sqlite3.Connection, pid: str, limit: int = 2) -> list[str]:
    rows = conn.execute(
        "SELECT summary FROM mistakes WHERE problem_id=? ORDER BY id DESC LIMIT ?",
        (pid, limit),
    ).fetchall()
    return [str(row["summary"]) for row in rows]
