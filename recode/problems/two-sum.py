# ---
# description: "Given an array of integers and a target, return indices of two numbers that add up to target"
# difficulty: easy
# tags: [arrays, hash-table]
# source: leetcode/1
# ---

SOLUTION = """
def two_sum(nums: list[int], target: int) -> list[int]:
    \"\"\"Return indices of two numbers that add up to target.\"\"\"
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
""".strip()

DESCRIPTION = "Given an array of integers and a target, return indices of two numbers that add up to target."

# ── Test cases ──

def _test_basic(ns):
    fn = ns.get("two_sum")
    assert fn is not None, "two_sum not found"
    result = fn([2, 7, 11, 15], 9)
    assert result == [0, 1], f"got {result}"

def _test_different_order(ns):
    fn = ns["two_sum"]
    result = fn([3, 2, 4], 6)
    assert sorted(result) == [1, 2], f"got {result}"

def _test_duplicates(ns):
    fn = ns["two_sum"]
    result = fn([3, 3], 6)
    assert sorted(result) == [0, 1], f"got {result}"

TEST_CASES = [_test_basic, _test_different_order, _test_duplicates]
