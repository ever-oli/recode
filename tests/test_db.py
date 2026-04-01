from __future__ import annotations

from db import get_db, get_row, log_mistake, recent_mistakes, reset_progress, sm2_update


def test_sm2_update_and_reset_progress(tmp_path):
    conn = get_db(tmp_path / "study.db")

    sm2_update(conn, "two-sum", 3)
    row = get_row(conn, "two-sum")

    assert row is not None
    assert row["reps"] == 1
    assert row["last_rating"] == 3

    log_mistake(conn, "two-sum", "missed hash map case")
    assert recent_mistakes(conn, "two-sum") == ["missed hash map case"]

    reset_progress(conn, "two-sum")
    assert get_row(conn, "two-sum") is None
    assert recent_mistakes(conn, "two-sum") == []
