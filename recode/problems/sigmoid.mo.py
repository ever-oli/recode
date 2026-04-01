"""
Sigmoid function — marimo notebook format for Recode.

This is a Recode problem with reactive test execution.
The user's code is injected into `user_attempt` and all test cells
re-run automatically when it changes.
"""
import marimo

app = marimo.App()


# === Reference Solution ===
@app.cell
def solution_cell():
    """Reference solution — what the student is trying to remember."""
    SOLUTION = '''
import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))
'''
    DESCRIPTION = "Implement the sigmoid activation function using NumPy."
    return SOLUTION, DESCRIPTION


# === User's Code ===
@app.cell
def user_code_cell(SOLUTION):
    """
    The user's attempt. At runtime, Recode overrides `user_attempt`
    with the student's code via app.run(defs={"user_attempt": user_code}).
    """
    user_attempt = SOLUTION  # default: reference solution
    return user_attempt,


# === Test Execution ===
@app.cell
def test_runner(user_attempt):
    """Execute user's code and run tests."""
    import numpy as np

    # Execute user's code in isolated namespace
    ns = {}
    try:
        exec(user_attempt, ns)
    except SyntaxError as e:
        test_results = [("syntax check", False, f"Syntax error: {e}")]
    except Exception as e:
        test_results = [("exec", False, f"Runtime error: {e}")]
    else:
        test_results = []
        sigmoid = ns.get("sigmoid")

        if sigmoid is None:
            test_results.append(("sigmoid function exists", False, "Function not found in code"))
        else:
            # Test 1: sigmoid(0) == 0.5
            try:
                out = sigmoid(0)
                ok = abs(float(out) - 0.5) < 1e-6
                test_results.append(("sigmoid(0) == 0.5", ok, f"got {out}"))
            except Exception as e:
                test_results.append(("sigmoid(0) == 0.5", False, str(e)))

            # Test 2: sigmoid(large positive) → 1
            try:
                out = float(sigmoid(1000))
                ok = abs(out - 1.0) < 1e-4
                test_results.append(("sigmoid(1000) ≈ 1.0", ok, f"got {out}"))
            except Exception as e:
                test_results.append(("sigmoid(1000) ≈ 1.0", False, str(e)))

            # Test 3: sigmoid(large negative) → 0
            try:
                out = float(sigmoid(-1000))
                ok = abs(out - 0.0) < 1e-4
                test_results.append(("sigmoid(-1000) ≈ 0.0", ok, f"got {out}"))
            except Exception as e:
                test_results.append(("sigmoid(-1000) ≈ 0.0", False, str(e)))

            # Test 4: symmetry: sigmoid(-x) = 1 - sigmoid(x)
            try:
                x = 2.5
                ok = abs(float(sigmoid(-x)) - (1 - float(sigmoid(x)))) < 1e-6
                test_results.append(("sigmoid(-x) == 1 - sigmoid(x)", ok, ""))
            except Exception as e:
                test_results.append(("sigmoid(-x) == 1 - sigmoid(x)", False, str(e)))

            # Test 5: vectorized input
            try:
                out = sigmoid(np.array([0, 1, -1]))
                ok = hasattr(out, "__len__") and len(out) == 3
                test_results.append(("accepts array input", ok, f"shape: {getattr(out, 'shape', 'N/A')}"))
            except Exception as e:
                test_results.append(("accepts array input", False, str(e)))

    return test_results,


# === Test Summary Display (marimo UI — optional) ===
@app.cell
def display_cell(test_results):
    """Show test results in the notebook UI when viewed in marimo."""
    import marimo as mo

    passed = sum(1 for _, p, _ in test_results if p)
    total = len(test_results)

    rows = []
    for name, passed_flag, detail in test_results:
        icon = "✅" if passed_flag else "❌"
        detail_str = f" — {detail}" if detail else ""
        rows.append(f"{icon} **{name}**{detail_str}")

    status = "🟢 All passed!" if passed == total else f"🔴 {passed}/{total} passed"
    mo.md(f"## Test Results\n{status}\n\n" + "\n".join(rows))
    return
