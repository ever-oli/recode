from __future__ import annotations

from recode import __version__
from recode.cli import main


def test_cli_version(capsys):
    assert main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == __version__


def test_cli_paths(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("RECODE_HOME", str(tmp_path))

    assert main(["--paths"]) == 0
    captured = capsys.readouterr()

    assert "problems_dir=" in captured.out
    assert "db_path=" in captured.out
