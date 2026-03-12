"""
test_runner.py — Execute problem test cases via marimo or direct exec.

Supports two test formats:
1. Legacy: TEST_CASES list in regular .py problems (exec-based)
2. Marimo: .mo.py marimo notebook problems with reactive test cells

Both produce a list of (name: str, passed: bool, detail: str) tuples
that the UI can display alongside the diff.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


def run_tests_exec(problem_path: Path, user_code: str) -> list[TestResult]:
    """
    Run tests from a regular .py problem file that defines TEST_CASES.
    
    TEST_CASES is a list of dicts:
        {"name": "sigmoid(0) == 0.5", "fn": lambda ns: abs(ns["sigmoid"](0) - 0.5) < 1e-6}
    Or a list of callables that raise AssertionError on failure:
        TEST_CASES = [test_sigmoid_zero, test_sigmoid_large]
    
    The test functions receive a namespace dict with the user's exec'd code.
    """
    results: list[TestResult] = []

    # Load the problem module to get TEST_CASES
    spec = importlib.util.spec_from_file_location("_prob", problem_path)
    if spec is None or spec.loader is None:
        return [TestResult("load problem", False, "Could not load problem file")]
    
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        return [TestResult("load problem", False, f"Import error: {e}")]

    test_cases = getattr(mod, "TEST_CASES", None)
    if not test_cases:
        return []  # No tests defined

    # Execute user's code in a clean namespace
    user_ns: dict = {}
    try:
        exec(user_code, user_ns)
    except Exception as e:
        return [TestResult("exec user code", False, f"Syntax/runtime error: {e}")]

    # Run each test
    for i, tc in enumerate(test_cases):
        if isinstance(tc, dict):
            name = tc.get("name", f"test_{i}")
            fn = tc.get("fn")
            if fn is None:
                results.append(TestResult(name, False, "No test function provided"))
                continue
            try:
                passed = fn(user_ns)
                results.append(TestResult(name, bool(passed), "" if passed else "assertion failed"))
            except AssertionError as e:
                results.append(TestResult(name, False, str(e)))
            except Exception as e:
                results.append(TestResult(name, False, f"{type(e).__name__}: {e}"))
        elif callable(tc):
            name = getattr(tc, "__name__", f"test_{i}")
            try:
                tc(user_ns)
                results.append(TestResult(name, True, ""))
            except AssertionError as e:
                results.append(TestResult(name, False, str(e)))
            except Exception as e:
                results.append(TestResult(name, False, f"{type(e).__name__}: {e}"))
        else:
            results.append(TestResult(f"test_{i}", False, f"Unknown test case type: {type(tc)}"))

    return results


def run_tests_marimo(notebook_path: Path, user_code: str) -> list[TestResult]:
    """
    Run tests from a marimo notebook (.mo.py) by injecting user_code
    and capturing test cell outputs.
    
    The notebook must define an `app` (marimo.App) with a `user_attempt`
    variable that defaults to the solution. We override it with user_code.
    """
    try:
        import marimo
    except ImportError:
        return [TestResult("marimo", False, "marimo not installed: pip install marimo")]

    # Import the notebook as a module to get the app
    spec = importlib.util.spec_from_file_location("_marimo_nb", notebook_path)
    if spec is None or spec.loader is None:
        return [TestResult("load notebook", False, "Could not load notebook file")]
    
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as e:
        return [TestResult("load notebook", False, f"Import error: {e}")]

    app = getattr(mod, "app", None)
    if app is None:
        return [TestResult("marimo app", False, "No marimo.App found in notebook")]

    # Run with user's code injected
    try:
        outputs, defs = app.run(defs={"user_attempt": user_code})
    except Exception as e:
        return [TestResult("marimo run", False, f"Execution error: {e}")]

    # Extract test results from defs
    # The notebook should define a `test_results` variable
    test_results = defs.get("test_results", [])
    
    if not test_results:
        # Fallback: check if there's a single pass/fail
        if "tests_passed" in defs:
            passed = defs["tests_passed"]
            return [TestResult("all tests", bool(passed), "")]
        return [TestResult("tests", False, "No test_results or tests_passed found in notebook")]

    # Convert to TestResult objects
    results = []
    for tr in test_results:
        if isinstance(tr, (list, tuple)) and len(tr) >= 2:
            results.append(TestResult(
                name=str(tr[0]),
                passed=bool(tr[1]),
                detail=str(tr[2]) if len(tr) > 2 else "",
            ))
        elif isinstance(tr, dict):
            results.append(TestResult(
                name=tr.get("name", "unnamed"),
                passed=bool(tr.get("passed", False)),
                detail=tr.get("detail", ""),
            ))
        else:
            results.append(TestResult("unknown", bool(tr), ""))

    return results


def run_tests(problem_path: Path, user_code: str) -> list[TestResult]:
    """
    Auto-detect test format and run appropriate test runner.
    
    - .mo.py files → marimo runner
    - .py files with TEST_CASES → exec runner
    - otherwise → no tests
    """
    if problem_path.suffix == ".py" and ".mo" in problem_path.stem:
        return run_tests_marimo(problem_path, user_code)
    elif problem_path.suffix == ".py":
        # Check if TEST_CASES is defined
        try:
            spec = importlib.util.spec_from_file_location("_prob_check", problem_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                if hasattr(mod, "TEST_CASES"):
                    return run_tests_exec(problem_path, user_code)
        except Exception:
            pass
    return []


def format_test_results(results: list[TestResult]) -> str:
    """Format test results for display in the TUI."""
    if not results:
        return ""
    
    lines = ["[bold]Test Results[/bold]", "─" * 40]
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    for r in results:
        icon = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        detail = f"  [dim]{r.detail}[/dim]" if r.detail else ""
        lines.append(f"  {icon} {r.name}{detail}")
    
    lines.append("─" * 40)
    color = "green" if passed == total else "yellow" if passed > 0 else "red"
    lines.append(f"[bold {color}]{passed}/{total} passed[/bold {color}]")
    
    return "\n".join(lines)


def has_tests(problem_path: Path) -> bool:
    """Check if a problem file has test cases defined."""
    if ".mo" in problem_path.stem:
        return True
    try:
        spec = importlib.util.spec_from_file_location("_prob_check", problem_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return hasattr(mod, "TEST_CASES")
    except Exception:
        pass
    return False
