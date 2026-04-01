from __future__ import annotations

from pathlib import Path

from recode import runtime


def test_prepare_runtime_uses_recode_home_and_seeds_problems(monkeypatch, tmp_path):
    monkeypatch.setenv("RECODE_HOME", str(tmp_path))
    monkeypatch.delenv("RECODE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("RECODE_DATA_DIR", raising=False)
    monkeypatch.delenv("RECODE_STATE_DIR", raising=False)
    monkeypatch.delenv("PROBLEMS_DIR", raising=False)
    monkeypatch.delenv("DB_PATH", raising=False)

    configured = runtime.prepare_runtime()

    assert configured.config_dir == (tmp_path / "config").resolve()
    assert configured.data_dir == (tmp_path / "data").resolve()
    assert configured.state_dir == (tmp_path / "state").resolve()
    assert configured.problems_dir.exists()
    assert configured.db_path.parent.exists()
    assert any(configured.problems_dir.rglob("*.py"))


def test_doctor_report_lists_key_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("RECODE_HOME", str(tmp_path))
    configured = runtime.prepare_runtime()

    report = runtime.doctor_report(configured)

    assert "recode 0.1.0" in report
    assert f"problems_dir={configured.problems_dir}" in report
    assert f"db_path={configured.db_path}" in report
