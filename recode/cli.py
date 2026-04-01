from __future__ import annotations

import argparse
from pathlib import Path

from recode import __version__
from recode.runtime import doctor_report, prepare_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recode",
        description="Terminal spaced repetition for coding problems.",
    )
    parser.add_argument("--version", action="store_true", help="print the installed recode version and exit")
    parser.add_argument("--paths", action="store_true", help="print resolved runtime paths and exit")
    parser.add_argument("--doctor", action="store_true", help="print runtime diagnostics and exit")
    parser.add_argument("--problems-dir", type=Path, help="override the writable problems directory")
    parser.add_argument("--db-path", type=Path, help="override the SQLite database path")
    parser.add_argument("--editor", help="override the editor command for this run")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    runtime = prepare_runtime(
        problems_dir=args.problems_dir,
        db_path=args.db_path,
        editor=args.editor,
    )

    if args.paths:
        print(f"config_dir={runtime.config_dir}")
        print(f"data_dir={runtime.data_dir}")
        print(f"state_dir={runtime.state_dir}")
        print(f"problems_dir={runtime.problems_dir}")
        print(f"bundled_problems_dir={runtime.bundled_problems_dir}")
        print(f"db_path={runtime.db_path}")
        return 0

    if args.doctor:
        print(doctor_report(runtime))
        return 0

    from app import main as app_main

    app_main(
        problems_dir=runtime.problems_dir,
        db_path=runtime.db_path,
        editor=runtime.editor,
    )
    return 0
