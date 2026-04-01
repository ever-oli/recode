SOLUTION = """
def flatten(lst):
    \"\"\"Flatten a nested list of arbitrary depth.\"\"\"
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
""".strip()

DESCRIPTION = "Implement a recursive function to flatten a nested list."

# ── Test cases ──

def _test_simple(ns):
    fn = ns.get("flatten")
    assert fn is not None, "flatten function not found"
    assert fn([1, [2, 3], 4]) == [1, 2, 3, 4], f"got {fn([1, [2, 3], 4])}"

def _test_deep(ns):
    fn = ns["flatten"]
    assert fn([1, [2, [3, [4]]]]) == [1, 2, 3, 4], "deeply nested failed"

def _test_empty(ns):
    fn = ns["flatten"]
    assert fn([]) == [], "empty list should return []"
    assert fn([[], [[]]]) == [], "nested empty lists"

def _test_mixed(ns):
    fn = ns["flatten"]
    assert fn([1, "a", [2, ["b", [3]]]]) == [1, "a", 2, "b", 3], "mixed types failed"

def _test_single(ns):
    fn = ns["flatten"]
    assert fn([1]) == [1], "single element"
    assert fn([[1]]) == [1], "single nested element"

TEST_CASES = [_test_simple, _test_deep, _test_empty, _test_mixed, _test_single]
