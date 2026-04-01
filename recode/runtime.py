from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_config_dir, user_data_dir, user_state_dir

from recode import __version__

APP_NAME = "recode"
APP_AUTHOR = "Ever"
CODE_EXTENSIONS = {".py", ".jl", ".R"}


@dataclass(frozen=True)
class RuntimePaths:
    config_dir: Path
    data_dir: Path
    state_dir: Path
    bundled_problems_dir: Path
    problems_dir: Path
    db_path: Path
    editor: str


_ACTIVE_RUNTIME: RuntimePaths | None = None


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _default_dirs() -> tuple[Path, Path, Path]:
    root_override = os.environ.get("RECODE_HOME")
    if root_override:
        root = Path(root_override).expanduser().resolve()
        return (root / "config", root / "data", root / "state")

    return (
        Path(user_config_dir(APP_NAME, APP_AUTHOR)).expanduser().resolve(),
        Path(user_data_dir(APP_NAME, APP_AUTHOR)).expanduser().resolve(),
        Path(user_state_dir(APP_NAME, APP_AUTHOR)).expanduser().resolve(),
    )


def _has_problem_files(path: Path) -> bool:
    if not path.exists():
        return False
    return any(p.is_file() and p.suffix in CODE_EXTENSIONS for p in path.rglob("*"))


def _seed_problem_bundle(source: Path, target: Path) -> None:
    if not source.exists() or _has_problem_files(target):
        return
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc"),
    )


def build_runtime(
    *,
    problems_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    editor: str | None = None,
) -> RuntimePaths:
    load_dotenv(override=False)

    default_config_dir, default_data_dir, default_state_dir = _default_dirs()

    config_dir = Path(os.environ.get("RECODE_CONFIG_DIR", default_config_dir)).expanduser().resolve()
    data_dir = Path(os.environ.get("RECODE_DATA_DIR", default_data_dir)).expanduser().resolve()
    state_dir = Path(os.environ.get("RECODE_STATE_DIR", default_state_dir)).expanduser().resolve()

    config_env = config_dir / ".env"
    if config_env.exists():
        load_dotenv(config_env, override=False)

    bundled_problems_dir = _package_root() / "problems"
    resolved_problems_dir = Path(
        problems_dir
        or os.environ.get("PROBLEMS_DIR")
        or data_dir / "problems"
    ).expanduser().resolve()
    resolved_db_path = Path(
        db_path
        or os.environ.get("DB_PATH")
        or state_dir / "study_data.db"
    ).expanduser().resolve()

    return RuntimePaths(
        config_dir=config_dir,
        data_dir=data_dir,
        state_dir=state_dir,
        bundled_problems_dir=bundled_problems_dir,
        problems_dir=resolved_problems_dir,
        db_path=resolved_db_path,
        editor=editor or os.environ.get("EDITOR", "hx"),
    )


def prepare_runtime(
    *,
    problems_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    editor: str | None = None,
) -> RuntimePaths:
    global _ACTIVE_RUNTIME

    runtime = build_runtime(problems_dir=problems_dir, db_path=db_path, editor=editor)
    runtime.config_dir.mkdir(parents=True, exist_ok=True)
    runtime.data_dir.mkdir(parents=True, exist_ok=True)
    runtime.state_dir.mkdir(parents=True, exist_ok=True)
    runtime.db_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.problems_dir.parent.mkdir(parents=True, exist_ok=True)
    _seed_problem_bundle(runtime.bundled_problems_dir, runtime.problems_dir)

    _ACTIVE_RUNTIME = runtime
    return runtime


def get_runtime() -> RuntimePaths:
    return _ACTIVE_RUNTIME or prepare_runtime()


def doctor_report(runtime: RuntimePaths | None = None) -> str:
    active = runtime or get_runtime()
    checks = {
        "Gemini key": bool(os.environ.get("GEMINI_API_KEY")),
        "OpenRouter key": bool(os.environ.get("OPENROUTER_API_KEY")),
        "OpenCode CLI": shutil.which("opencode") is not None,
        "Editor": shutil.which(active.editor) is not None,
    }

    lines = [
        f"recode {__version__}",
        f"config_dir={active.config_dir}",
        f"data_dir={active.data_dir}",
        f"state_dir={active.state_dir}",
        f"problems_dir={active.problems_dir}",
        f"bundled_problems_dir={active.bundled_problems_dir}",
        f"db_path={active.db_path}",
        f"editor={active.editor}",
    ]
    for label, ok in checks.items():
        lines.append(f"{label}={'ok' if ok else 'missing'}")
    return "\n".join(lines)
