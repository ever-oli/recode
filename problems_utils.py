"""
problems.py — Problem discovery, metadata loading, diff rendering, and helpers.
"""
from __future__ import annotations

import difflib
import importlib.util
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from rich.table import Table
from rich.text import Text

# Supported problem file extensions
CODE_EXTENSIONS = {".py", ".jl", ".R"}

# Marimo notebook detection: files with .mo.py or .mo.jl etc. in stem
MARIMO_MARKER = ".mo"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Parse YAML-like frontmatter from the start of a file.
    
    Format:
        # ---
        # key: value
        # tags: [a, b, c]
        # ---
        <rest of file>
    
    Returns (metadata_dict, remaining_text).
    Falls back to ({}, text) if no frontmatter found.
    """
    lines = text.splitlines()
    
    # Check for frontmatter start (# --- or ---)
    if not lines:
        return {}, text
    
    first = lines[0].strip()
    if first not in ("# ---", "---"):
        return {}, text
    
    # Determine comment style
    is_commented = first.startswith("#")
    delimiter = "# ---" if is_commented else "---"
    
    # Find the closing ---
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == delimiter:
            end_idx = i
            break
    
    if end_idx is None:
        return {}, text
    
    # Parse metadata lines
    meta = {}
    for line in lines[1:end_idx]:
        stripped = line.strip()
        if is_commented:
            stripped = re.sub(r'^#\s*', '', stripped)
        
        if not stripped or ':' not in stripped:
            continue
        
        key, _, value = stripped.partition(':')
        key = key.strip()
        value = value.strip()
        
        # Parse lists: [a, b, c]
        if value.startswith('[') and value.endswith(']'):
            items = [v.strip().strip('"').strip("'") for v in value[1:-1].split(',')]
            meta[key] = [v for v in items if v]
        # Parse quoted strings
        elif value.startswith('"') and value.endswith('"'):
            meta[key] = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            meta[key] = value[1:-1]
        # Parse booleans
        elif value.lower() in ('true', 'yes'):
            meta[key] = True
        elif value.lower() in ('false', 'no'):
            meta[key] = False
        # Parse numbers
        else:
            try:
                if '.' in value:
                    meta[key] = float(value)
                else:
                    meta[key] = int(value)
            except ValueError:
                meta[key] = value
    
    # Remaining text after frontmatter
    remaining = '\n'.join(lines[end_idx + 1:])
    return meta, remaining


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
    """Load metadata from a problem file.
    
    Supports:
    - YAML frontmatter (# --- ... # ---) with description, difficulty, tags, source, etc.
    - Legacy SOLUTION/DESCRIPTION variables in .py files
    - Raw text for non-Python files
    
    Returns dict with keys: solution, description, difficulty, tags, source, prerequisites, 
    and any other frontmatter fields.
    """
    raw_text = path.read_text()
    
    # Parse frontmatter
    meta, remaining = parse_frontmatter(raw_text)
    
    # Non-Python files: use frontmatter + raw text
    if path.suffix != ".py":
        result = {
            "solution": remaining.strip() if remaining.strip() else raw_text,
            "description": meta.get("description", ""),
            "difficulty": meta.get("difficulty", ""),
            "tags": meta.get("tags", []),
            "source": meta.get("source", ""),
            "prerequisites": meta.get("prerequisites", []),
        }
        # Include any extra frontmatter fields
        for k, v in meta.items():
            if k not in result:
                result[k] = v
        return result
    
    # Python files: try to import for SOLUTION/DESCRIPTION (legacy)
    # Use remaining text (after frontmatter) for import
    spec = importlib.util.spec_from_file_location("_prob", path)
    if spec is None or spec.loader is None:
        return {"solution": raw_text, "description": meta.get("description", "")}
    
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        pass
    
    # Frontmatter takes precedence, fallback to module globals, then raw text
    solution = getattr(mod, "SOLUTION", remaining.strip() or raw_text)
    description = meta.get("description", "") or getattr(mod, "DESCRIPTION", "")
    
    result = {
        "solution": solution,
        "description": description,
        "difficulty": meta.get("difficulty", getattr(mod, "DIFFICULTY", "")),
        "tags": meta.get("tags", getattr(mod, "TAGS", [])),
        "source": meta.get("source", getattr(mod, "SOURCE", "")),
        "prerequisites": meta.get("prerequisites", []),
    }
    # Include any extra frontmatter fields
    for k, v in meta.items():
        if k not in result:
            result[k] = v
    return result


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


def is_marimo_problem(path: Path) -> bool:
    """Check if a problem file is a marimo notebook (.mo.py, .mo.jl, etc.)."""
    return MARIMO_MARKER in path.stem


def has_test_cases(problem_path: Path) -> bool:
    """
    Check if a problem has test cases defined.
    Works for both marimo notebooks and regular .py files with TEST_CASES.
    """
    if is_marimo_problem(problem_path):
        return True
    
    if problem_path.suffix != ".py":
        return False
    
    try:
        spec = importlib.util.spec_from_file_location("_prob_check", problem_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return hasattr(mod, "TEST_CASES")
    except Exception:
        pass
    return False


def problem_badges(path: Path) -> list[tuple[str, str]]:
    """
    Return display badges for a problem (icon, tooltip).
    E.g., [("🧪", "has tests"), ("📓", "marimo notebook"), ("⭐", "medium")]
    """
    badges = []
    if is_marimo_problem(path):
        badges.append(("📓", "marimo notebook"))
    if has_test_cases(path):
        badges.append(("🧪", "has tests"))
    if path.suffix == ".jl":
        badges.append(("🟣", "Julia"))
    elif path.suffix == ".R":
        badges.append(("🔵", "R"))
    
    # Add difficulty badge from metadata
    meta = load_problem_meta(path)
    difficulty = meta.get("difficulty", "")
    if difficulty:
        diff_icons = {"easy": ("🟢", "easy"), "medium": ("🟡", "medium"), "hard": ("🔴", "hard")}
        if difficulty.lower() in diff_icons:
            badges.append(diff_icons[difficulty.lower()])
    
    # Add source badge
    source = meta.get("source", "")
    if source:
        if source.startswith("leetcode"):
            badges.append(("🟧", "LeetCode"))
        elif source.startswith("exercism"):
            badges.append(("🟦", "Exercism"))
        elif source.startswith("hf://") or "huggingface" in source:
            badges.append(("🤗", "HuggingFace"))
    
    return badges
