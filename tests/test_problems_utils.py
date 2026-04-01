from __future__ import annotations

from pathlib import Path

from problems_utils import get_problem_id, load_problem_meta, scan_collections, scan_problems


def test_scan_and_load_problem_meta(tmp_path):
    root = tmp_path / "problems"
    root.mkdir()
    nested = root / "generated"
    nested.mkdir()

    problem = nested / "adder.py"
    problem.write_text(
        '# ---\n'
        '# description: "Add two numbers."\n'
        '# difficulty: easy\n'
        '# tags: [arrays, math]\n'
        '# ---\n'
        "SOLUTION = '''\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "'''.strip()\n"
    )

    assert scan_problems(nested) == [problem]
    collections = scan_collections(root)
    assert root not in collections
    assert nested in collections
    assert get_problem_id(problem, root) == "generated/adder"

    meta = load_problem_meta(problem)
    assert meta["description"] == "Add two numbers."
    assert meta["difficulty"] == "easy"
    assert meta["tags"] == ["arrays", "math"]
    assert "def add" in meta["solution"]
