"""
ai.py — AI provider calls, prompt builders, and LaTeX-to-Unicode rendering.
"""
from __future__ import annotations

import difflib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from base64 import b64encode
from urllib.parse import urlparse


# ── Config (read at import time so callers don't need to pass env vars) ────────
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL",     "gemini-2.0-flash")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
AI_PROVIDER      = os.environ.get("AI_PROVIDER",      "openrouter")
OPENCODE_SERVER  = os.environ.get("OPENCODE_SERVER_URL", "http://127.0.0.1:4096")
OPENCODE_AUTOSTART = os.environ.get("OPENCODE_AUTOSTART", "1") not in {"0", "false", "False"}

_OPENCODE_PROCESS: subprocess.Popen | None = None


# ── LaTeX → Unicode ────────────────────────────────────────────────────────────
def render_latex(text: str) -> str:
    """
    Convert inline LaTeX math in AI responses to readable Unicode.
    Handles $...$, $$...$$, \\(...\\), and \\[...\\] delimiters.
    Falls back gracefully if pylatexenc is not installed.
    """
    try:
        import re
        from pylatexenc.latex2text import LatexNodes2Text
        conv = LatexNodes2Text(math_mode="text")

        def _replace(m: re.Match) -> str:
            inner = m.group(1) or m.group(2) or m.group(3)
            try:
                return conv.latex_to_text(inner)
            except Exception:
                return m.group(0)

        text = re.sub(r'\$\$(.+?)\$\$', _replace, text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$',     _replace, text, flags=re.DOTALL)
        text = re.sub(r'\\\((.+?)\\\)', _replace, text, flags=re.DOTALL)
        text = re.sub(r'\\\[(.+?)\\\]', _replace, text, flags=re.DOTALL)
    except ImportError:
        pass
    return text


# ── Providers ─────────────────────────────────────────────────────────────────
def _gemini(prompt: str) -> str:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No GEMINI_API_KEY found in .env."
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (result.text or "").strip()
    except Exception as e:
        return f"**Gemini error:** {e}"


def _openrouter(prompt: str) -> str:
    import json
    import urllib.request
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No OPENROUTER_API_KEY found in .env."
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/ever-oli/codi",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"**OpenRouter error:** {e}"


def ai_call(prompt: str) -> str:
    if AI_PROVIDER == "gemini":
        return _gemini(prompt)
    return _openrouter(prompt)

# ── Helpers ───────────────────────────────────────────────────────────────
_EXT_TO_LANG = {
    ".py": "python", ".jl": "julia", ".R": "r",
}

def _lang_for(name: str) -> str:
    """Infer code-fence language from a problem filename."""
    import os
    _, ext = os.path.splitext(name)
    return _EXT_TO_LANG.get(ext, "python")

def _opencode_request(method: str, path: str, body: dict | None = None, timeout: int = 60) -> tuple[dict, bool]:
    payload = None if body is None else json.dumps(body).encode()
    url = f"{OPENCODE_SERVER.rstrip('/')}{path}"
    headers = {"Content-Type": "application/json"}

    password = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if password:
        username = os.environ.get("OPENCODE_SERVER_USERNAME", "opencode")
        token = b64encode(f"{username}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"

    req = urllib.request.Request(url, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode().strip()
            return (json.loads(raw) if raw else {}, False)
    except urllib.error.URLError as first_error:
        if _try_autostart_opencode_server():
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode().strip()
                return (json.loads(raw) if raw else {}, True)
        raise first_error


def _try_autostart_opencode_server() -> bool:
    global _OPENCODE_PROCESS
    if not OPENCODE_AUTOSTART:
        return False

    parsed = urlparse(OPENCODE_SERVER)
    host = parsed.hostname or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return False

    if _opencode_health_ok():
        return True

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    if _OPENCODE_PROCESS is None or _OPENCODE_PROCESS.poll() is not None:
        try:
            _OPENCODE_PROCESS = subprocess.Popen(
                ["opencode", "serve", "--hostname", host, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False

    for _ in range(30):
        if _opencode_health_ok():
            return True
        time.sleep(0.2)
    return False


def _opencode_health_ok() -> bool:
    try:
        req = urllib.request.Request(f"{OPENCODE_SERVER.rstrip('/')}/global/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode() or "{}")
            return bool(data.get("healthy"))
    except Exception:
        return False


def opencode_chat_health() -> tuple[str, str]:
    """Return a human-readable OpenCode connectivity report and status."""
    parsed = urlparse(OPENCODE_SERVER)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    auth_mode = "basic auth" if os.environ.get("OPENCODE_SERVER_PASSWORD") else "none"
    autostart_mode = "enabled" if OPENCODE_AUTOSTART else "disabled"
    healthy = _opencode_health_ok()

    report = (
        "OpenCode diagnostics\n"
        f"- server: {OPENCODE_SERVER}\n"
        f"- host: {host}\n"
        f"- port: {port}\n"
        f"- auth: {auth_mode}\n"
        f"- autostart: {autostart_mode}\n"
        f"- health: {'ok' if healthy else 'unreachable'}"
    )
    return (report, "connected" if healthy else "offline")


def _extract_text(parts: list[dict]) -> str:
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        value = part.get("text") or part.get("content") or part.get("value")
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    return "\n\n".join(chunks).strip()


def opencode_chat(problem_name: str, ref_code: str, user_code: str, message: str, session_id: str | None) -> tuple[str, str, str]:
    """
    Send a chat message through a local OpenCode server.
    Returns: (assistant_reply, session_id)
    """
    try:
        sid = session_id
        autostarted = False
        if not sid:
            created, did_autostart = _opencode_request("POST", "/session", {"title": f"Codi chat: {problem_name}"})
            autostarted = autostarted or did_autostart
            sid = created.get("id")
            if not sid:
                return ("OpenCode server responded, but no session id was returned.", "", "offline")

            context = (
                "You are Codi's in-app coding study chat. "
                "Keep answers concise, practical, and focused on learning. "
                "Do not reveal the full reference solution unless explicitly asked.\n\n"
                f"Problem: {problem_name}\n\n"
                "Reference solution:\n"
                f"```python\n{ref_code}\n```\n\n"
                "Current user attempt:\n"
                f"```python\n{user_code or '# empty'}\n```"
            )
            _, did_autostart = _opencode_request(
                "POST",
                f"/session/{sid}/message",
                {
                    "noReply": True,
                    "parts": [{"type": "text", "text": context}],
                },
            )
            autostarted = autostarted or did_autostart

        result, did_autostart = _opencode_request(
            "POST",
            f"/session/{sid}/message",
            {"parts": [{"type": "text", "text": message}]},
        )
        autostarted = autostarted or did_autostart
        parts = result.get("parts", [])
        text = _extract_text(parts)
        if not text:
            text = "I received your message, but I couldn't parse a text response from OpenCode."
        status = "autostarted" if autostarted else "connected"
        return (text, sid, status)
    except urllib.error.URLError:
        return (
            "Could not reach OpenCode server. Codi tried to auto-start it, but could not. "
            "Start it with `opencode serve --port 4096`, ensure `opencode` is on PATH, "
            "or set `OPENCODE_SERVER_URL`.",
            "",
            "offline",
        )
    except Exception as e:
        return (f"OpenCode chat error: {e}", session_id or "", "offline")


# ── Prompts ───────────────────────────────────────────────────────────────────
def get_hint(problem_name: str, ref_code: str, user_code: str | None) -> str:
    has_attempt = bool(user_code and user_code.strip())
    lang = _lang_for(problem_name)
    if has_attempt:
        diff_text = "\n".join(difflib.unified_diff(
            ref_code.splitlines(), (user_code or "").splitlines(),
            fromfile="reference", tofile="yours", lineterm="",
        ))
        prompt = f"""\
You are a coding tutor helping a student study ML implementations from memory.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

The student's current attempt diff vs reference:
```diff
{diff_text}
```

Give a single Socratic hint that nudges them toward what they're missing \
without revealing the answer. Be concise (2-4 sentences max). \
Focus on the most important gap in their attempt."""
    else:
        prompt = f"""\
You are a coding tutor helping a student study ML implementations from memory.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

The student hasn't written anything yet. Give a single Socratic hint to \
help them get started — what is the core concept or structure they need to \
think about? Be concise (2-4 sentences max). Do not give away the answer."""
    return ai_call(prompt)


def get_suggest_fix(problem_name: str, ref_code: str, user_code: str) -> str:
    lang = _lang_for(problem_name)
    diff_text = "\n".join(difflib.unified_diff(
        ref_code.splitlines(), user_code.splitlines(),
        fromfile="reference", tofile="yours", lineterm="",
    ))
    prompt = f"""\
You are a coding tutor reviewing a student's ML implementation attempt.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

Student's attempt diff vs reference:
```diff
{diff_text}
```

Be direct and specific. List exactly what is wrong and what needs to change \
to match the reference. Use short bullet points. Do not be vague. \
Do not explain concepts they already got right."""
    return ai_call(prompt)


def get_explain(problem_name: str, ref_code: str) -> str:
    lang = _lang_for(problem_name)
    prompt = f"""\
You are a coding tutor explaining an ML concept to a student who just finished \
(or attempted) an implementation exercise.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

Give a clear, concise explanation (5-8 sentences) of:
1. What this component does conceptually in the broader ML context
2. Why each key design decision in the implementation exists
3. One common real-world mistake or misconception to watch out for

Write for someone who can code but is still building intuition. \
Do not just restate the code line by line."""
    return ai_call(prompt)
