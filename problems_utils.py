"""
problems.py — Problem discovery, metadata loading, diff rendering, and helpers.
"""
from __future__ import annotations

import difflib
import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

from rich.table import Table
from rich.text import Text

# Supported problem file extensions
CODE_EXTENSIONS = {".py", ".jl", ".R"}


def _is_code_file(p: Path) -> bool:
    return p.is_file() and p.suffix in CODE_EXTENSIONS


def scan_problems(problems_dir: Path) -> list[Path]:
    if not problems_dir.exists():
        return []
    return sorted(p for p in problems_dir.iterdir() if _is_code_file(p))


def scan_collections(root: Path) -> list[Path]:
    """
    Recursively find all directories in `root` that contain at least one code file.
    Includes `root` itself if it contains code files.
    """
    if not root.exists():
        return []

    collections = set()

    # Check root itself first
    if any(_is_code_file(p) for p in root.iterdir()):
        collections.add(root)

    # Walk directory tree
    for path in root.rglob("*"):
        if path.is_dir():
            has_code = any(_is_code_file(p) for p in path.iterdir())
            if has_code:
                collections.add(path)

    # Sort by path depth then name for nice display order
    return sorted(list(collections), key=lambda p: (len(p.parts), p.name))


def get_problem_id(problem_path: Path, root: Path) -> str:
    """
    Get a stable ID for the problem.
    - If direct child of root: use filename stem (backward compatibility)
    - Else: use relative path string without extension
    """
    try:
        rel = problem_path.relative_to(root)
    except ValueError:
        # Fallback if somehow path is not relative to root
        return problem_path.stem

    if len(rel.parts) == 1:
        return problem_path.stem

    # Use forward slashes for ID regardless of OS
    return str(rel.with_suffix("")).replace("\\", "/")


def load_problem_meta(path: Path) -> dict:
    """Load SOLUTION and DESCRIPTION from a problem file.

    For .py files, tries to import and read SOLUTION/DESCRIPTION variables.
    For other file types (.jl, .R, etc.), reads raw text as the solution.
    """
    # Non-Python files: just read raw text
    if path.suffix != ".py":
        return {"solution": path.read_text(), "description": ""}

    spec = importlib.util.spec_from_file_location("_prob", path)
    if spec is None or spec.loader is None:
        return {"solution": path.read_text(), "description": ""}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        pass
    return {
        "solution":    getattr(mod, "SOLUTION",    path.read_text()),
        "description": getattr(mod, "DESCRIPTION", ""),
    }


def build_side_by_side(ref_code: str, user_code: str) -> Table:
    """Side-by-side diff with soft block and intraline highlighting."""
    ref_lines  = ref_code.splitlines()
    user_lines = user_code.splitlines()

    def line_text(line: str, base_style: str, spans: list[tuple[int, int]], span_style: str) -> Text:
        text = Text(line if line else " ", style=base_style)
        for start, end in spans:
            if start < end:
                text.stylize(span_style, start, end)
        return text

    def intraline_spans(left: str, right: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
        left_spans: list[tuple[int, int]] = []
        right_spans: list[tuple[int, int]] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag in ("replace", "delete") and i1 != i2:
                left_spans.append((i1, i2))
            if tag in ("replace", "insert") and j1 != j2:
                right_spans.append((j1, j2))
        return left_spans, right_spans

    table = Table(
        show_header=True,
        header_style="bold #9ecbff",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("ref", justify="right", style="dim", width=5, no_wrap=True)
    table.add_column("reference", no_wrap=False, ratio=1)
    table.add_column("you", justify="right", style="dim", width=5, no_wrap=True)
    table.add_column("yours", no_wrap=False, ratio=1)

    matcher = difflib.SequenceMatcher(None, ref_lines, user_lines, autojunk=False)
    ref_no = 1
    user_no = 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for rl, ul in zip(ref_lines[i1:i2], user_lines[j1:j2]):
                left = line_text(rl, "grey70", [], "")
                right = line_text(ul, "grey70", [], "")
                table.add_row(str(ref_no), left, str(user_no), right)
                ref_no += 1
                user_no += 1
        elif tag == "replace":
            ref_chunk  = ref_lines[i1:i2]
            user_chunk = user_lines[j1:j2]
            length = max(len(ref_chunk), len(user_chunk))
            for idx in range(length):
                rl = ref_chunk[idx] if idx < len(ref_chunk) else None
                ul = user_chunk[idx] if idx < len(user_chunk) else None

                if rl is not None and ul is not None:
                    left_spans, right_spans = intraline_spans(rl, ul)
                    left = line_text(rl, "white on #2f1414", left_spans, "bold white on #7a1f1f")
                    right = line_text(ul, "white on #12321f", right_spans, "bold white on #1f7a3a")
                    table.add_row(str(ref_no), left, str(user_no), right)
                    ref_no += 1
                    user_no += 1
                elif rl is not None:
                    left = line_text(rl, "white on #2f1414", [], "")
                    table.add_row(str(ref_no), left, "", Text(" "))
                    ref_no += 1
                elif ul is not None:
                    right = line_text(ul, "white on #12321f", [], "")
                    table.add_row("", Text(" "), str(user_no), right)
                    user_no += 1
        elif tag == "delete":
            for line in ref_lines[i1:i2]:
                left = line_text(line, "white on #2f1414", [], "")
                table.add_row(str(ref_no), left, "", Text(" "))
                ref_no += 1
        elif tag == "insert":
            for line in user_lines[j1:j2]:
                right = line_text(line, "white on #12321f", [], "")
                table.add_row("", Text(" "), str(user_no), right)
                user_no += 1

    return table


def status_label(row: sqlite3.Row | None) -> tuple[str, str]:
    if row is None:
        return "New", "white"
    nxt = datetime.fromisoformat(row["next_review"])
    if datetime.now() >= nxt:
        return "Due", "bold red"
    diff = nxt - datetime.now()
    return (f"In {diff.days}d", "dim") if diff.days >= 1 else ("Due soon", "bold red")


def max_rating_for(attempts: int) -> int:
    if attempts <= 1: return 4
    if attempts == 2: return 3
    if attempts == 3: return 2
    return 1
