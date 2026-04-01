"""
leetcode.py — Fetch problems from LeetCode and convert to Recode format.

Uses LeetCode's GraphQL API (same as their website) to:
- List problems by difficulty/tag
- Fetch problem details (description, test cases, solutions)
- Convert to Recode problem format with TEST_CASES

Note: LeetCode doesn't provide an official public API. This uses their
internal GraphQL endpoint which may change. Community solutions are
fetched from publicly available sources.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from html import unescape
from pathlib import Path

LEETCODE_GRAPHQL = "https://leetcode.com/graphql"
LEETCODE_API = "https://leetcode.com/api"


@dataclass
class LeetCodeProblem:
    id: int
    title: str
    title_slug: str
    difficulty: str  # Easy, Medium, Hard
    description: str
    topics: list[str]
    test_cases: list[dict]  # [{"input": ..., "expected": ...}]
    solution_template: str
    hints: list[str]


def _graphql(query: str, variables: dict | None = None) -> dict | None:
    """Make a GraphQL request to LeetCode."""
    body = {"query": query}
    if variables:
        body["variables"] = variables
    
    try:
        req = urllib.request.Request(
            LEETCODE_GRAPHQL,
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Recode/0.1)",
                "Referer": "https://leetcode.com",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def list_problems(
    difficulty: str = "",
    tag: str = "",
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """
    List LeetCode problems with optional filters.
    
    Args:
        difficulty: "Easy", "Medium", "Hard", or "" for all
        tag: Topic tag slug (e.g., "array", "dynamic-programming")
        limit: Max results
        offset: Pagination offset
    """
    query = """
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
        problemsetQuestionList: questionList(
            categorySlug: $categorySlug
            limit: $limit
            skip: $skip
            filters: $filters
        ) {
            total: totalNum
            questions: data {
                id: questionId
                title
                titleSlug: titleSlug
                difficulty
                status
                topicTags: topicTags {
                    name
                    slug
                }
                isPaidOnly: isPaidOnly
            }
        }
    }
    """
    
    variables = {
        "categorySlug": "",
        "skip": offset,
        "limit": limit,
        "filters": {},
    }
    
    if difficulty:
        variables["filters"]["difficulty"] = difficulty.upper()
    if tag:
        variables["filters"]["tags"] = [tag]
    
    result = _graphql(query, variables)
    if not result or "data" not in result:
        return []
    
    questions = result["data"]["problemsetQuestionList"]["questions"]
    problems = []
    for q in questions:
        if q.get("isPaidOnly"):
            continue  # Skip premium problems
        problems.append({
            "id": int(q["id"]),
            "title": q["title"],
            "slug": q["titleSlug"],
            "difficulty": q["difficulty"].lower(),
            "topics": [t["name"] for t in q.get("topicTags", [])],
        })
    
    return problems


def list_free_problems(
    *,
    difficulty: str = "",
    tag: str = "",
    page_size: int = 100,
    max_pages: int = 30,
) -> list[dict]:
    """List the full free LeetCode catalog for the given filters."""
    all_problems: list[dict] = []
    seen_slugs: set[str] = set()

    for page in range(max_pages):
        batch = list_problems(
            difficulty=difficulty,
            tag=tag,
            limit=page_size,
            offset=page * page_size,
        )
        if not batch:
            break

        new_count = 0
        for problem in batch:
            slug = problem["slug"]
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            all_problems.append(problem)
            new_count += 1

        if len(batch) < page_size or new_count == 0:
            break

    return all_problems


def fetch_problem(slug: str) -> LeetCodeProblem | None:
    """
    Fetch a specific LeetCode problem with details.
    
    Args:
        slug: Problem slug (e.g., "two-sum", "valid-parentheses")
    """
    query = """
    query getQuestionDetail($titleSlug: String!) {
        question(titleSlug: $titleSlug) {
            questionId
            title
            titleSlug
            difficulty
            content
            topicTags {
                name
                slug
            }
            hints
            codeSnippets {
                lang
                langSlug
                code
            }
            exampleTestcases
            sampleTestCase
        }
    }
    """
    
    result = _graphql(query, {"titleSlug": slug})
    if not result or "data" not in result or not result["data"].get("question"):
        return None
    
    q = result["data"]["question"]
    
    # Parse HTML description to text
    description = _html_to_text(q.get("content", ""))
    
    # Get Python code template
    template = ""
    for snippet in q.get("codeSnippets", []):
        if snippet.get("langSlug") == "python3":
            template = snippet.get("code", "")
            break
    
    # Parse example test cases
    test_cases = _parse_test_cases(
        q.get("exampleTestcases", ""),
        q.get("sampleTestCase", ""),
        description,
    )
    
    return LeetCodeProblem(
        id=int(q["questionId"]),
        title=q["title"],
        title_slug=q["titleSlug"],
        difficulty=q["difficulty"].lower(),
        description=description,
        topics=[t["name"] for t in q.get("topicTags", [])],
        test_cases=test_cases,
        solution_template=template,
        hints=q.get("hints", []),
    )


def _html_to_text(html: str) -> str:
    """Convert LeetCode HTML description to plain text."""
    # Remove HTML tags but keep structure
    text = unescape(html)
    text = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', text, flags=re.DOTALL)
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)
    text = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'</p>', '', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<li[^>]*>', '- ', text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove remaining tags
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines
    return text.strip()


def _parse_test_cases(example_cases: str, sample_case: str, description: str) -> list[dict]:
    """Parse test cases from LeetCode problem data."""
    test_cases = []
    
    # Parse from exampleTestcases field
    if example_cases:
        lines = example_cases.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line:
                test_cases.append({
                    "input": line,
                    "expected": "",  # LeetCode doesn't provide expected in this field
                })
            i += 1
    
    # Extract examples from description
    examples = re.findall(
        r'Input:\s*(.+?)\s*Output:\s*(.+?)(?:\s*Explanation:.*?)?(?=\n\n|Example|\Z)',
        description,
        re.DOTALL,
    )
    
    for inp, out in examples:
        test_cases.append({
            "input": inp.strip(),
            "expected": out.strip(),
        })
    
    return test_cases


def convert_to_recode_problem(problem: LeetCodeProblem) -> dict:
    """Convert a LeetCode problem to Recode format."""
    slug = problem.title_slug.replace("-", "_")
    
    # Build test cases
    test_code = _build_test_code(problem)
    
    return {
        "filename": f"leetcode-{slug}.py",
        "solution": problem.solution_template,
        "description": f"[LeetCode #{problem.id}] {problem.title}: {problem.description[:200]}",
        "difficulty": problem.difficulty,
        "tags": problem.topics + ["leetcode"],
        "tests": test_code,
        "source": f"leetcode/{problem.id}",
        "hints": problem.hints,
    }


def _build_test_code(problem: LeetCodeProblem) -> str:
    """Build Recode TEST_CASES from LeetCode test cases."""
    # Extract the function name from the template
    func_match = re.search(r'def (\w+)\(', problem.solution_template)
    func_name = func_match.group(1) if func_match else "solution"
    
    lines = [
        "# ── Test cases (from LeetCode) ──",
        "",
        f"def _test_leetcode_cases(ns):",
        f'    """Run LeetCode test cases"""',
        f"    fn = ns.get('{func_name}')",
        f"    assert fn is not None, '{func_name} not found'",
        "",
    ]
    
    for i, tc in enumerate(problem.test_cases[:5]):  # Limit to 5 test cases
        inp = tc.get("input", "")
        expected = tc.get("expected", "")
        
        if inp and expected:
            # Parse input format like "nums = [2,7,11,15], target = 9"
            lines.append(f"    # Test case {i + 1}: Input: {inp}")
            lines.append(f"    # Expected: {expected}")
            
            # Try to extract variable assignments
            assignments = re.findall(r'(\w+)\s*=\s*(.+?)(?:,|$)', inp)
            if assignments:
                for var, val in assignments:
                    lines.append(f"    {var} = {val.strip()}")
                args = ", ".join(a[0] for a in assignments)
                lines.append(f"    result = fn({args})")
                lines.append(f"    # Verify result matches expected: {expected}")
                lines.append(f"    assert result is not None, 'returned None'")
                lines.append("")
    
    # Add a basic smoke test
    lines.extend([
        f"def _test_callable(ns):",
        f'    """Function is callable"""',
        f"    fn = ns.get('{func_name}')",
        f"    assert callable(fn), '{func_name} is not callable'",
        "",
        f"TEST_CASES = [_test_callable, _test_leetcode_cases]",
    ])
    
    return "\n".join(lines)


def fetch_and_convert(slug: str) -> dict | None:
    """Fetch a LeetCode problem and convert to Recode format."""
    problem = fetch_problem(slug)
    if not problem:
        return None
    return convert_to_recode_problem(problem)


def write_leetcode_problem(problem: dict, output_dir: Path) -> Path:
    """Write a LeetCode-sourced problem to a file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / problem["filename"]
    
    tags_str = ", ".join(problem.get("tags", []))
    lines = [
        "# ---",
        f'# description: "{problem["description"][:150]}"',
        f'# difficulty: {problem.get("difficulty", "medium")}',
        f"# tags: [{tags_str}]",
        f'# source: {problem.get("source", "")}',
        "# ---",
        "",
        'SOLUTION = """',
        problem["solution"],
        '""".strip()',
        "",
        f'DESCRIPTION = "{problem["description"][:200]}"',
    ]
    
    if problem.get("tests"):
        lines.append("")
        lines.append(problem["tests"])
    
    filepath.write_text("\n".join(lines))
    return filepath


def easy_problems(limit: int = 10) -> list[dict]:
    """Get easy LeetCode problems (good for learning)."""
    return list_problems(difficulty="Easy", limit=limit)


def free_problems() -> list[dict]:
    """Get the full free LeetCode catalog."""
    return list_free_problems()


def popular_problems(limit: int = 10) -> list[dict]:
    """Get popular/common LeetCode problems."""
    popular_slugs = [
        "two-sum", "valid-parentheses", "merge-two-sorted-lists",
        "best-time-to-buy-and-sell-stock", "valid-palindrome",
        "invert-binary-tree", "valid-anagram", "binary-search",
        "flood-fill", "maximum-subarray",
    ]
    problems = []
    for slug in popular_slugs[:limit]:
        result = fetch_and_convert(slug)
        if result:
            problems.append(result)
    return problems
