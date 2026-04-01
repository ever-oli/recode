"""
paper_generator.py — Generate Recode problems from arXiv papers.

Uses the arXiv API for metadata + AI to generate implementation exercises.
Supports simple mode (whole paper) and detailed mode (section picker).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


@dataclass
class PaperInfo:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published: str
    pdf_url: str
    abs_url: str


def parse_arxiv_url(url_or_id: str) -> str:
    """
    Extract arXiv ID from various URL formats or bare ID.
    
    Accepts:
    - https://arxiv.org/abs/2402.03300
    - https://arxiv.org/pdf/2402.03300
    - arxiv:2402.03300
    - 2402.03300
    """
    # Try URL patterns
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
        r'arxiv:(\d+\.\d+)',
        r'^(\d+\.\d+)$',
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id.strip())
        if m:
            return m.group(1)
    return url_or_id.strip()


def fetch_paper(arxiv_id: str) -> PaperInfo | None:
    """Fetch paper metadata from arXiv API."""
    arxiv_id = parse_arxiv_url(arxiv_id)
    url = f"{ARXIV_API}?id_list={arxiv_id}"
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Recode/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            xml_data = resp.read().decode()
    except Exception:
        return None
    
    root = ET.fromstring(xml_data)
    entry = root.find("a:entry", NS)
    if entry is None:
        return None
    
    title = (entry.find("a:title", NS).text or "").strip().replace("\n", " ")
    authors = [a.find("a:name", NS).text for a in entry.findall("a:author", NS)]
    abstract = (entry.find("a:summary", NS).text or "").strip()
    categories = [c.get("term") for c in entry.findall("a:category", NS) if c.get("term")]
    published = (entry.find("a:published", NS).text or "")[:10]
    
    return PaperInfo(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        categories=categories,
        published=published,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def extract_sections(paper: PaperInfo) -> list[str]:
    """
    Extract section headings from a paper's abstract and content.
    Since we can't easily parse PDF in-process, we derive likely sections
    from the paper's category and abstract structure.
    
    For full content, users can use fetch_content on the PDF URL.
    """
    sections = []
    
    # Common ML paper sections based on abstract structure
    abstract_lower = paper.abstract.lower()
    
    section_hints = [
        ("Introduction", ["introduce", "propose", "motivation", "we present"]),
        ("Method", ["method", "approach", "framework", "algorithm", "architecture", "propose"]),
        ("Attention Mechanism", ["attention", "self-attention", "multi-head", "query", "key", "value"]),
        ("Training", ["train", "optimiz", "loss function", "gradient", "backprop"]),
        ("Architecture", ["network", "layer", "block", "module", "encoder", "decoder"]),
        ("Evaluation", ["experiment", "evaluate", "benchmark", "result", "performance"]),
        ("Mathematical Foundations", ["equation", "formulation", "theorem", "proof"]),
    ]
    
    for section_name, keywords in section_hints:
        if any(kw in abstract_lower for kw in keywords):
            sections.append(section_name)
    
    return sections or ["Full Paper"]


def generate_problems_from_paper(
    paper: PaperInfo,
    sections: list[str] | None = None,
    num_problems: int = 3,
    language: str = "python",
    use_marimo: bool = True,
) -> list[dict]:
    """
    Use AI to generate implementation exercises from a paper.
    
    Returns list of dicts with keys: filename, description, solution, tests
    """
    from ai import ai_call
    
    section_text = ""
    if sections:
        section_text = f"\nFocus on these aspects: {', '.join(sections)}"
    
    test_instruction = ""
    if use_marimo:
        test_instruction = """
Also include a TEST_CASES list with 2-4 test functions per problem. Each test function 
should accept a namespace dict and raise AssertionError on failure. Name them clearly 
like _test_basic_forward, _test_shape_output, etc.
"""
    
    lang_examples = {
        "python": "Use NumPy for numerical operations. Use clear variable names.",
        "julia": "Use standard Julia arrays and broadcasting. Use clear variable names.",
    }
    lang_note = lang_examples.get(language, lang_examples["python"])
    
    prompt = f"""\
You are an ML educator creating coding exercises from a research paper.

Paper: {paper.title}
Authors: {', '.join(paper.authors[:3])}
arXiv: {paper.arxiv_id}

Abstract:
{paper.abstract}
{section_text}

Generate {num_problems} implementation exercises that help someone deeply understand 
this paper by coding its key components from memory.

{lang_note}

For each problem, provide:
1. filename: snake_case, descriptive (e.g., "multi-head-attention.py")
2. description: 1-2 sentence prompt for the student (what to implement)
3. solution: complete, working {language} code (the reference implementation)
4. difficulty: easy / medium / hard
{test_instruction}
Return the response as a JSON array. Example format:
[
  {{
    "filename": "attention-mechanism.py",
    "description": "Implement scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V",
    "solution": "import numpy as np\\n\\ndef scaled_dot_product_attention(Q, K, V):\\n    ...",
    "difficulty": "medium",
    "tests": [
      {{"name": "output shape", "fn_body": "fn = ns['scaled_dot_product_attention']\\nQ = np.random.randn(2, 4, 8)\\nK = np.random.randn(2, 4, 8)\\nV = np.random.randn(2, 4, 8)\\nout = fn(Q, K, V)\\nassert out.shape == (2, 4, 8), f'got {{out.shape}}'"}}
    ]
  }}
]

Return ONLY the JSON array, no other text."""
    
    response = ai_call(prompt)
    
    # Extract JSON from response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        return []
    
    try:
        problems = json.loads(json_match.group(0))
        return problems
    except json.JSONDecodeError:
        return []


def write_problem_file(
    problem: dict,
    output_dir: Path,
    paper: PaperInfo,
    use_marimo: bool = False,
) -> Path:
    """
    Write a generated problem to a file in Recode format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = problem.get("filename", "generated-problem.py")
    if not any(filename.endswith(ext) for ext in [".py", ".jl", ".R"]):
        filename += ".py"
    
    filepath = output_dir / filename
    solution = problem.get("solution", "")
    description = problem.get("description", "")
    difficulty = problem.get("difficulty", "medium")
    tests = problem.get("tests", [])
    
    if use_marimo and tests:
        content = _format_marimo_problem(solution, description, tests, paper)
    else:
        content = _format_standard_problem(solution, description, tests, paper, difficulty)
    
    filepath.write_text(content)
    return filepath


def _format_standard_problem(
    solution: str, description: str, tests: list, paper: PaperInfo, difficulty: str
) -> str:
    """Format as a standard .py problem file."""
    lines = []
    
    # Header comment
    lines.append(f'"""')
    lines.append(f'Generated from: {paper.title}')
    lines.append(f'arXiv: {paper.arxiv_id}')
    lines.append(f'Difficulty: {difficulty}')
    lines.append(f'"""')
    lines.append("")
    
    # Solution
    escaped_solution = solution.replace('"""', '\\"\\"\\"')
    lines.append(f'SOLUTION = """')
    lines.append(escaped_solution)
    lines.append(f'""".strip()')
    lines.append("")
    
    # Description
    escaped_desc = description.replace('"', '\\"')
    lines.append(f'DESCRIPTION = "{escaped_desc}"')
    
    # Tests (if provided)
    if tests:
        lines.append("")
        lines.append("# ── Test cases ──")
        lines.append("")
        for i, test in enumerate(tests):
            fn_name = f'_test_{test.get("name", f"test_{i}").replace(" ", "_").lower()}'
            fn_body = test.get("fn_body", "")
            lines.append(f'def {fn_name}(ns):')
            lines.append(f'    """{test.get("name", f"test_{i}")}"""')
            for line in fn_body.split("\n"):
                lines.append(f'    {line}')
            lines.append("")
        
        test_names = [f'_test_{t.get("name", f"test_{i}").replace(" ", "_").lower()}' 
                      for i, t in enumerate(tests)]
        lines.append(f'TEST_CASES = [{", ".join(test_names)}]')
    
    return "\n".join(lines)


def _format_marimo_problem(
    solution: str, description: str, tests: list, paper: PaperInfo
) -> str:
    """Format as a marimo notebook problem."""
    # For marimo format, we generate the test execution cell
    test_checks = []
    for i, test in enumerate(tests):
        fn_body = test.get("fn_body", "").replace('"', '\\"')
        name = test.get("name", f"test_{i}")
        test_checks.append(f'''
            # Test: {name}
            try:
{chr(10).join("                " + line for line in test.get("fn_body", "").split(chr(10)))}
                test_results.append(("{name}", True, ""))
            except AssertionError as e:
                test_results.append(("{name}", False, str(e)))
            except Exception as e:
                test_results.append(("{name}", False, f"{{type(e).__name__}}: {{e}}"))''')
    
    return f'''"""
{description}

Generated from: {paper.title}
arXiv: {paper.arxiv_id}
"""
import marimo

app = marimo.App()


@app.cell
def solution_cell():
    SOLUTION = """
{solution}
""".strip()
    DESCRIPTION = "{description}"
    return SOLUTION, DESCRIPTION


@app.cell
def user_code_cell(SOLUTION):
    user_attempt = SOLUTION
    return user_attempt,


@app.cell
def test_runner(user_attempt):
    test_results = []
    
    ns = {{}}
    try:
        exec(user_attempt, ns)
    except SyntaxError as e:
        test_results.append(("syntax", False, f"Syntax error: {{e}}"))
    except Exception as e:
        test_results.append(("exec", False, f"Runtime error: {{e}}"))
    else:
{chr(10).join(test_checks)}
    
    return test_results,
'''


def generate_problems(
    arxiv_url: str,
    output_dir: Path,
    sections: list[str] | None = None,
    num_problems: int = 3,
    language: str = "python",
    use_marimo: bool = True,
) -> tuple[PaperInfo | None, list[Path]]:
    """
    Main entry point: fetch paper, generate problems, write files.
    
    Returns (paper_info, list_of_written_files).
    """
    arxiv_id = parse_arxiv_url(arxiv_url)
    paper = fetch_paper(arxiv_id)
    if not paper:
        return None, []
    
    problems = generate_problems_from_paper(
        paper, sections=sections, num_problems=num_problems,
        language=language, use_marimo=use_marimo,
    )
    
    written = []
    for prob in problems:
        filepath = write_problem_file(prob, output_dir, paper, use_marimo=use_marimo)
        written.append(filepath)
    
    return paper, written
