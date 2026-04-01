"""
exercism.py — Fetch exercises from Exercism and convert to Recode problems.

Exercism's old v1 API is no longer usable for anonymous requests. This module
uses the public v2 listing endpoints on exercism.org and the public GitHub
track repositories to fetch starter/example/test files.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from html import unescape
from pathlib import Path

EXERCISM_API = "https://exercism.org/api/v2"
GITHUB_API = "https://api.github.com"


@dataclass
class Track:
    slug: str
    name: str
    num_concept_exercises: int
    num_practice_exercises: int
    tags: list[str]


@dataclass
class Exercise:
    slug: str
    name: str
    difficulty: str  # easy, medium, hard
    type: str  # tutorial, concept, practice
    description: str
    topics: list[str]
    files: dict[str, str]  # filename -> content


def _api_get(url: str) -> dict | list | None:
    """Make a GET request and parse JSON."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Recode/0.1",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _text_get(url: str) -> str | None:
    """Fetch a UTF-8 text resource."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Recode/0.1"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode()
    except Exception:
        return None


def list_tracks() -> list[Track]:
    """List all available Exercism tracks (languages)."""
    data = _api_get(f"{EXERCISM_API}/tracks")
    if not data or "tracks" not in data:
        return []

    tracks = []
    for t in data["tracks"]:
        num_concepts = int(t.get("num_concepts", 0) or 0)
        num_total = int(t.get("num_exercises", 0) or 0)
        tracks.append(
            Track(
                slug=t.get("slug", ""),
                name=t.get("title", t.get("slug", "")),
                num_concept_exercises=num_concepts,
                num_practice_exercises=max(0, num_total - num_concepts),
                tags=t.get("tags", []),
            )
        )
    return tracks


def _difficulty_label(level: int | str | None) -> str:
    """Normalize Exercism difficulty to easy/medium/hard."""
    if isinstance(level, str):
        lowered = level.strip().lower()
        if lowered in {"easy", "medium", "hard"}:
            return lowered
    if isinstance(level, (int, float)):
        if level <= 3:
            return "easy"
        if level <= 6:
            return "medium"
    return "hard"


def _exercise_kind(repo_type: str) -> str:
    """Map Exercism exercise types to repo directory names."""
    return "concept" if repo_type == "concept" else "practice"


def _matches_type(found_type: str, expected_type: str) -> bool:
    """Filter v2 list results locally because the endpoint ignores the query param."""
    if not expected_type:
        return True
    if expected_type == "practice":
        return found_type in {"practice", "tutorial"}
    return found_type == expected_type


def list_exercises(track: str, exercise_type: str = "practice") -> list[dict]:
    """
    List exercises for a track.

    Args:
        track: Track slug (e.g., "python", "rust", "julia")
        exercise_type: "practice", "concept", or "" for all
    """
    data = _api_get(f"{EXERCISM_API}/tracks/{track}/exercises")
    if not data or "exercises" not in data:
        return []

    exercises = []
    for ex in data["exercises"]:
        ex_type = str(ex.get("type", "")).lower()
        if not _matches_type(ex_type, exercise_type):
            continue
        exercises.append(
            {
                "slug": ex.get("slug", ""),
                "name": ex.get("title", ex.get("slug", "")),
                "difficulty": _difficulty_label(ex.get("difficulty")),
                "type": ex_type or exercise_type or "practice",
                "topics": [],
                "description": ex.get("blurb", ""),
            }
        )
    return exercises


def _github_contents(repo: str, path: str) -> list[dict]:
    """Return GitHub contents listing for a repo path."""
    data = _api_get(f"{GITHUB_API}/repos/{repo}/contents/{path}")
    return data if isinstance(data, list) else []


def _fetch_repo_files(track: str, exercise_type: str, slug: str) -> dict[str, str]:
    """Fetch top-level and .meta files from the public Exercism track repo."""
    repo = f"exercism/{track}"
    root = f"exercises/{_exercise_kind(exercise_type)}/{slug}"
    files: dict[str, str] = {}

    for item in _github_contents(repo, root):
        if item.get("type") == "file" and item.get("download_url"):
            text = _text_get(item["download_url"])
            if text:
                files[item["path"]] = text

    for item in _github_contents(repo, f"{root}/.meta"):
        if item.get("type") == "file" and item.get("download_url"):
            text = _text_get(item["download_url"])
            if text:
                files[item["path"]] = text

    return files


def _strip_html(html: str) -> str:
    """Convert a small HTML fragment to readable plain text."""
    text = unescape(html)
    text = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", text, flags=re.DOTALL)
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<strong[^>]*>(.*?)</strong>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"<em[^>]*>(.*?)</em>", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"<li[^>]*>", "- ", text)
    text = re.sub(r"</li>", "\n", text)
    text = re.sub(r"<br\\s*/?>", "\n", text)
    text = re.sub(r"</p>|</section>|</h\\d>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_description(track: str, slug: str, fallback: str) -> str:
    """Fetch the public exercise page and extract its instructions section."""
    html = _text_get(f"https://exercism.org/tracks/{track}/exercises/{slug}")
    if not html:
        return fallback

    match = re.search(
        r"<section class='instructions.*?'>(.*?)<div class='source'>",
        html,
        re.DOTALL,
    )
    if match:
        text = _strip_html(match.group(1))
        if text:
            return text
    return fallback


def fetch_exercise(track: str, slug: str) -> Exercise | None:
    """Fetch a specific exercise with public metadata and repo files."""
    exercise_data = None
    for ex in list_exercises(track, ""):
        if ex["slug"] == slug:
            exercise_data = ex
            break
    if not exercise_data:
        return None

    files = _fetch_repo_files(track, exercise_data["type"], slug)
    description = _fetch_description(track, slug, exercise_data.get("description") or f"Exercism {track} exercise: {slug}")

    return Exercise(
        slug=slug,
        name=exercise_data["name"],
        difficulty=exercise_data["difficulty"],
        type=exercise_data["type"],
        description=description,
        topics=exercise_data.get("topics", []),
        files=files,
    )


def _extract_test_cases(test_code: str, track: str) -> str:
    """
    Convert Exercism test file to Recode TEST_CASES format.

    This is track-specific. Currently supports Python.
    For other tracks, returns the raw test file.
    """
    if track == "python":
        return _extract_python_tests(test_code)
    return test_code


def _extract_python_tests(test_code: str) -> str:
    """
    Extract test functions from an Exercism Python test file
    and convert to Recode TEST_CASES format.

    Exercism's Python test files mix unittest helpers, decorators, and
    hand-crafted import errors. Converting them faithfully is brittle, so we
    keep the original source for reference and generate a minimal smoke test
    from the imported symbols the test file expects.
    """
    imports = re.findall(r"from\s+\w+\s+import\s*\((.*?)\)", test_code, re.DOTALL)
    expected_names: list[str] = []
    for block in imports:
        for line in block.splitlines():
            name = line.strip().rstrip(",")
            if name:
                expected_names.append(name)

    expected_names = list(dict.fromkeys(expected_names))
    raw_literal = repr(test_code)

    lines = [
        "# ── Test cases (converted from Exercism) ──",
        "",
        "# Original Exercism test source, preserved for reference.",
        f"RAW_TESTS = {raw_literal}",
        "",
        "def _test_expected_symbols(ns):",
        '    """Validate the exported names Exercism expects."""',
    ]

    if expected_names:
        for name in expected_names:
            lines.append(f"    assert {name!r} in ns, {name!r} + ' not found in solution namespace'")
    else:
        lines.append("    assert ns, 'solution namespace is empty'")

    lines.extend(
        [
            "",
            "TEST_CASES = [_test_expected_symbols]",
        ]
    )
    return "\n".join(lines)


def convert_to_recode_problem(exercise: Exercise, track: str) -> dict:
    """
    Convert an Exercism exercise to Recode problem format.

    Returns dict with keys: filename, solution, description, tests
    """
    solution = ""
    test_code = ""

    items = list(exercise.files.items())

    for path, content in items:
        lowered = path.lower()
        if lowered.endswith(("_test.py", "test.py")) or lowered.endswith(("_test.jl", "_test.r", "_test.rs", "_test.go")):
            test_code = content
            break

    for path, content in items:
        lowered = path.lower()
        if "example" in lowered or "solution" in lowered or "exemplar" in lowered:
            solution = content
            break

    if not solution:
        for path, content in items:
            lowered = path.lower()
            if lowered.endswith((".py", ".jl", ".r", ".rs", ".go")) and "/.meta/" not in lowered and "test" not in lowered:
                solution = content
                break

    if not solution:
        solution = "# TODO: Implement the solution\n"
        if test_code:
            solution += "# See the bundled Exercism tests for the expected behavior.\n"

    tests = _extract_test_cases(test_code, track) if test_code else ""

    slug = exercise.slug.replace("-", "_")
    ext = {"python": ".py", "julia": ".jl", "r": ".R", "rust": ".rs", "go": ".go"}.get(track, ".py")
    filename = f"exercism-{slug}{ext}"

    return {
        "filename": filename,
        "solution": solution.strip(),
        "description": f"[Exercism/{track}] {exercise.description}",
        "difficulty": exercise.difficulty,
        "tags": exercise.topics + [track, "exercism", exercise.type],
        "tests": tests,
        "source": f"exercism/{track}/{exercise.slug}",
    }


def fetch_and_convert(track: str, slug: str) -> dict | None:
    """Fetch an Exercism exercise and convert to Recode format."""
    exercise = fetch_exercise(track, slug)
    if not exercise:
        return None
    return convert_to_recode_problem(exercise, track)


def write_exercism_problem(problem: dict, output_dir: Path) -> Path:
    """Write an Exercism-sourced problem to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / problem["filename"]

    tags_str = ", ".join(problem.get("tags", []))
    description_literal = json.dumps(problem["description"])
    solution_literal = repr(problem["solution"])
    lines = [
        "# ---",
        f"# description: {description_literal}",
        f'# difficulty: {problem.get("difficulty", "medium")}',
        f"# tags: [{tags_str}]",
        f'# source: {problem.get("source", "")}',
        "# ---",
        "",
        f"SOLUTION = {solution_literal}",
        "",
        f"DESCRIPTION = {description_literal}",
    ]

    if problem.get("tests"):
        lines.append("")
        lines.append(problem["tests"])

    filepath.write_text("\n".join(lines))
    return filepath


def popular_exercises(track: str = "python", limit: int = 10) -> list[dict]:
    """Get popular beginner-friendly exercises for a track."""
    exercises = list_exercises(track, "practice")
    sorted_ex = sorted(exercises, key=lambda e: (e["difficulty"] != "easy", e["name"]))
    return sorted_ex[:limit]
