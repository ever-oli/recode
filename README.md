# Recode

A terminal-based spaced repetition tool for practicing any code from memory. Write implementations, compare against reference solutions with a side-by-side diff, get AI hints when stuck, and let the SM-2 algorithm schedule your reviews.

![Menu screen](screenshots/menu.png)

---

## Features

- **Spaced repetition** — SM-2 algorithm schedules reviews so you spend time on what you actually need to practice
- **Side-by-side diff** — your attempt vs. the reference solution, line by line
- **AI hints** — Socratic nudges that guide you without just giving away the answer
- **AI suggest fix** — targeted bullet-point suggestions when you want more direct feedback
- **OpenCode chat modal** — conversational help in-context while you study a problem
- **Agent-like chat tools** — presets (`/nudge`, `/test-me`), TODO capture, diff split view, and `/health`
- **158 themes** — full terminal.sexy palette, switchable live from the command palette
- **Extensible** — add any problem by dropping a `.py`, `.jl`, or `.R` file into your writable `problems_dir`

---

## Screenshots

| | |
|---|---|
| ![Problem list](screenshots/menu.png) | ![Diff view](screenshots/diff.png) |
| Problem list with SM-2 schedule | Side-by-side diff |
| ![Rating modal](screenshots/rating.png) | ![AI hint](screenshots/hint.png) |
| SM-2 rating (Again / Hard / Good / Easy) | AI Socratic hint |

![AI suggest fix](screenshots/suggest-fix.png)
*AI suggest fix — targeted bullet-point suggestions*

---

## Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (recommended), `pipx`, or `pip`
- A Gemini or OpenRouter API key
- [OpenCode CLI](https://opencode.ai/) on your PATH for in-app chat (auto-started by Recode)

---

## Install

### Homebrew

```bash
brew tap ever-oli/homebrew-tap
brew install ever-oli/homebrew-tap/recode
```

### PyPI

```bash
uv tool install recode-cli
# or
pipx install recode-cli
```

### Local dev

```bash
git clone https://github.com/ever-oli/recode.git
cd recode
uv sync
uv run python -m recode
```

---

## Configuration

Recode reads environment variables from your shell, a local `.env`, or `~/.config/recode/.env`.
Use `recode --paths` to print the exact runtime directories for your machine.

On first run, Recode seeds its bundled problem set into your writable `problems_dir` so generated and imported problems live beside the defaults instead of inside the installed package.

Example `.env`:

```env
# Option A — Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here
AI_PROVIDER=gemini

# Option B — OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
AI_PROVIDER=openrouter

# Optional: override the editor (default: hx)
EDITOR=nvim

# Optional: override runtime locations
# PROBLEMS_DIR=/path/to/your/problems
# DB_PATH=/path/to/recode.db

# Optional: OpenCode server URL for in-app chat modal
# OPENCODE_SERVER_URL=http://127.0.0.1:4096

# Optional: disable automatic `opencode serve` on first chat request
# OPENCODE_AUTOSTART=0
```

---

## Run

Once installed:

```bash
recode
```

Useful non-interactive commands:

```bash
recode --version
recode --paths
recode --doctor
```

### Chat modal behavior (OpenCode)

By default, pressing `c` in a problem will auto-start a local OpenCode server if it is not already running.
The chat modal shows a live status line (`connected`, `auto-started OpenCode`, or `offline`).
In chat, type `/health` to print OpenCode connection diagnostics.
Use `/help` for chat commands and `/clear` to clear chat history in the modal.
Chat also supports `/hint`, `/fix`, `/todo ...`, preset prompts, and `/split` (or `Ctrl+D`) to show diff + chat side by side.

If you prefer running it yourself (or disabled auto-start), start it manually:

```bash
opencode serve --port 4096
```

---

## Usage

| Key | Action |
|-----|--------|
| `Enter` | Open the selected problem |
| `/` | Focus search |
| `c` | Change collection on the main menu |
| `g` | Generate problems from an arXiv paper |
| `i` | Import from Exercism or LeetCode |
| `r` | Refresh the problem list |
| `q` / `Ctrl+C` | Quit |

Inside a problem:

| Key | Action |
|-----|--------|
| `e` | Open the editor |
| `s` | Submit and review |
| `h` | Ask for a hint |
| `f` | Ask for a suggested fix |
| `c` | Open the in-problem chat modal |
| `x` | Explain the solution or gap |
| `q` | Return to the menu |

### Workflow

1. Select a problem from the list and press `Enter`
2. Press `e` to open your editor and write the implementation from memory
3. Save and close the editor
4. Recode shows a side-by-side diff of your attempt vs. the reference
5. Use **Hint** or **Suggest Fix** if you need AI assistance
6. Rate your recall (Again / Hard / Good / Easy)
7. SM-2 schedules the next review automatically

---

## Adding Problems

Problems are plain `.py`, `.jl`, or `.R` files. Drop them into the writable `problems_dir` from `recode --paths` and they will appear in the list on the next refresh (`r`).
`TensorPoly` ships as a bundled collection and is copied into your writable problems directory on first run. The built-in importer currently supports Exercism Python and free LeetCode problems.

A problem file contains two things:

```python
# The reference solution Recode compares against
SOLUTION = """
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
""".strip()

# The prompt shown in the problem list
DESCRIPTION = "Implement the sigmoid function using NumPy."
```

36 curated ML problems are included covering ResNet, Transformers, GANs, U-Net, ViT, and more.

---

## License

MIT
