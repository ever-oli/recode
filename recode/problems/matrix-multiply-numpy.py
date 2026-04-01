SOLUTION = """
import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    \"\"\"Multiply two matrices using NumPy.\"\"\"
    return np.matmul(A, B)
""".strip()

DESCRIPTION = "Implement matrix multiplication using NumPy's matmul."

# ── Test cases (exec-based, works without marimo) ──
# Each test receives the user's exec'd namespace and should
# raise AssertionError on failure or return True/False.

def _test_basic_mult(ns):
    """2x2 * 2x2"""
    fn = ns.get("matrix_multiply")
    assert fn is not None, "matrix_multiply not found"
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = fn(A, B)
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(result, expected), f"got {result}, expected {expected}"

def _test_identity(ns):
    """A * I = A"""
    fn = ns["matrix_multiply"]
    A = np.array([[1, 2, 3], [4, 5, 6]])
    I = np.eye(3)
    result = fn(A, I)
    assert np.allclose(result, A), "A * I should equal A"

def _test_rectangular(ns):
    """3x2 * 2x4"""
    fn = ns["matrix_multiply"]
    A = np.random.randn(3, 2)
    B = np.random.randn(2, 4)
    result = fn(A, B)
    assert result.shape == (3, 4), f"wrong shape: {result.shape}"
    assert np.allclose(result, np.matmul(A, B))

def _test_type_error(ns):
    """Incompatible shapes should raise"""
    fn = ns["matrix_multiply"]
    A = np.array([[1, 2]])
    B = np.array([[1, 2, 3]])
    try:
        fn(A, B)
        assert False, "should have raised an error for incompatible shapes"
    except (ValueError, RuntimeError):
        pass  # expected

TEST_CASES = [_test_basic_mult, _test_identity, _test_rectangular, _test_type_error]
