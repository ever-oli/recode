"""
modals.py — Reusable Textual modal screens: AIModal, ConfirmModal, RatingModal.
"""
from __future__ import annotations

import threading
from pathlib import Path
from datetime import datetime
from typing import Callable

from rich.panel import Panel
from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Markdown as MarkdownWidget, RichLog, Static

from recode.runtime import get_runtime

RATING_LABELS = {1: "Again", 2: "Hard", 3: "Good", 4: "Easy"}
RATING_DESC   = {
    1: "forgot it completely",
    2: "got it with real effort",
    3: "got it with some hesitation",
    4: "recalled perfectly",
}


class AIModal(ModalScreen):
    """Generic modal that fires an AI fetch in a background thread."""
    BINDINGS = [Binding("escape,q", "dismiss", "Close")]

    def __init__(self, title: str, fetch_fn) -> None:
        super().__init__()
        self._title    = title
        self._fetch_fn = fetch_fn

    def compose(self):
        yield Vertical(
            Label(f"[bold]{escape(self._title)}[/]", id="hint-title"),
            MarkdownWidget("*asking Recode...*", id="hint-md"),
            id="hint-box"
        )

    def on_mount(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from ai import render_latex
        text = render_latex(self._fetch_fn())
        self.app.call_from_thread(self._display, text)

    def _display(self, text: str) -> None:
        self.query_one("#hint-md", MarkdownWidget).update(text)


class ChatModal(ModalScreen):
    """Interactive chat modal backed by OpenCode session messages."""

    BINDINGS = [
        Binding("escape,q", "dismiss", "Close"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+k", "focus_input", "Focus input"),
        Binding("ctrl+d", "toggle_split", "Split"),
    ]

    def __init__(
        self,
        title: str,
        send_fn: Callable[[str], tuple[str, str] | str],
        context_fn: Callable[[], str] | None = None,
        mistakes_fn: Callable[[], list[str]] | None = None,
        diff_fn: Callable[[], str] | None = None,
        open_hint_fn: Callable[[], None] | None = None,
        open_fix_fn: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._send_fn = send_fn
        self._context_fn = context_fn
        self._mistakes_fn = mistakes_fn
        self._diff_fn = diff_fn
        self._open_hint_fn = open_hint_fn
        self._open_fix_fn = open_fix_fn
        self._busy = False
        self._split = False
        self._todos: list[dict[str, str | bool]] = []

    def compose(self):
        with Vertical(id="chat-box"):
            with Horizontal(id="chat-header"):
                yield Label(f"[bold]{escape(self._title)}[/]", id="chat-title")
                yield Label("[dim]status: ready[/]", id="chat-status")
            yield Label("[dim]context: loading...[/]", id="chat-context")
            with Horizontal(id="chat-main"):
                yield RichLog(id="chat-log", markup=True, highlight=False, wrap=True, auto_scroll=True)
                yield Static("", id="chat-diff", classes="hidden")
            yield Input(placeholder="Ask Recode chat...", id="chat-input")
            yield Label("[dim]Enter send | /nudge /explain-gap /test-me /checklist | /hint /fix | /todo ... | /split[/]", id="chat-help")

    def on_mount(self) -> None:
        self._refresh_context()
        self._append_system("Recode chat is ready. Ask about this problem, your diff, or the concept.")
        mistakes = self._mistakes_fn() if self._mistakes_fn else []
        if mistakes:
            self._append_system("Recent mistakes:")
            for idx, item in enumerate(mistakes, start=1):
                self._append_system(f"  {idx}. {item}")
        self.query_one("#chat-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if self._busy or not text:
            return
        event.input.value = ""

        lowered = text.lower()
        if lowered == "/clear":
            self.action_clear_chat()
            return
        if lowered == "/help":
            self._append_system("/health diagnostics | /hint open hint modal | /fix open suggest-fix modal")
            self._append_system("/todo add <text> | /todo | /todo done <n> | /split toggles chat/diff split")
            return
        if lowered == "/hint":
            if self._open_hint_fn:
                self._open_hint_fn()
                self._append_system("Opened Hint modal.")
            else:
                self._append_system("Hint action is unavailable in this context.")
            return
        if lowered == "/fix":
            if self._open_fix_fn:
                self._open_fix_fn()
                self._append_system("Opened Suggest Fix modal.")
            else:
                self._append_system("Suggest Fix action is unavailable in this context.")
            return
        if lowered == "/split":
            self.action_toggle_split()
            return
        if lowered == "/diff":
            self._show_diff_preview()
            return
        if lowered.startswith("/todo"):
            self._handle_todo(text)
            return

        presets = {
            "/nudge": "Give me one concise Socratic nudge based on my current attempt and do not reveal the answer.",
            "/explain-gap": "Explain the single most important gap in my current attempt in 3 short bullet points.",
            "/test-me": "Quiz me with 3 short questions about this problem, one at a time, and wait after each one.",
            "/checklist": "Give me a short implementation checklist I can follow before I submit.",
        }
        if lowered in presets:
            text = presets[lowered]
            self._append_system(f"preset: {lowered}")

        self._append_message("you", text)
        self._set_status("thinking")
        self._set_busy(True)
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def action_clear_chat(self) -> None:
        if self._busy:
            return
        self.query_one("#chat-log", RichLog).clear()
        self._append_system("Chat cleared. Start a new thread with your next question.")

    def action_focus_input(self) -> None:
        self.query_one("#chat-input", Input).focus()

    def action_toggle_split(self) -> None:
        self._split = not self._split
        pane = self.query_one("#chat-diff", Static)
        if self._split:
            pane.remove_class("hidden")
            self._show_diff_preview()
        else:
            pane.add_class("hidden")
            pane.update("")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        inp = self.query_one("#chat-input", Input)
        inp.disabled = busy
        if busy:
            inp.placeholder = "OpenCode is thinking…"
        else:
            inp.placeholder = "Ask Recode chat..."
            inp.focus()

    def _run(self, text: str) -> None:
        try:
            result = self._send_fn(text)
            if isinstance(result, tuple):
                answer = str(result[0])
                status = str(result[1]) if len(result) > 1 else "connected"
            else:
                answer = str(result)
                status = "connected"
        except Exception as e:
            answer = f"Chat error: {e}"
            status = "offline"
        self.app.call_from_thread(self._display, answer, status)

    def _display(self, text: str, status: str) -> None:
        self._append_message("recode", text)
        self._set_status(status)
        self._set_busy(False)
        self._refresh_context()

    def _append_system(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(f"[dim]{escape(text)}[/]")

    def _append_message(self, role: str, text: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        if role == "you":
            title = " YOU "
            style = "cyan"
        else:
            title = " OPENCODE "
            style = "green"

        body = Text(text, style="white")
        panel = Panel(
            body,
            title=title,
            subtitle=now,
            border_style=style,
            expand=True,
        )
        self.query_one("#chat-log", RichLog).write(panel)

    def _set_status(self, status: str) -> None:
        if status == "connected":
            label = "[green]status: connected[/]"
        elif status == "autostarted":
            label = "[yellow]status: auto-started OpenCode[/]"
        elif status == "thinking":
            label = "[cyan]status: waiting for reply[/]"
        elif status == "offline":
            label = "[red]status: offline[/]"
        else:
            label = f"[dim]status: {escape(status)}[/]"
        self.query_one("#chat-status", Label).update(label)

    def _refresh_context(self) -> None:
        if not self._context_fn:
            return
        self.query_one("#chat-context", Label).update(self._context_fn())

    def _show_diff_preview(self) -> None:
        if not self._diff_fn:
            self._append_system("No diff preview callback is available.")
            return
        text = self._diff_fn().strip()
        pane = self.query_one("#chat-diff", Static)
        if not text:
            pane.update("No diff available yet. Edit and save first.")
        else:
            pane.update(Syntax(text, "diff", theme="ansi_dark", word_wrap=False))
        if not self._split:
            self._append_system("Use /split (or Ctrl+D) to show diff beside chat.")

    def _handle_todo(self, command: str) -> None:
        parts = command.split(maxsplit=2)
        if len(parts) == 1:
            if not self._todos:
                self._append_system("TODO list is empty. Add one with /todo add <task>.")
                return
            self._append_system("TODO list:")
            for idx, item in enumerate(self._todos, start=1):
                mark = "x" if item["done"] else " "
                self._append_system(f"  {idx}. [{mark}] {item['text']}")
            return

        action = parts[1].lower()
        if action == "add":
            if len(parts) < 3 or not parts[2].strip():
                self._append_system("Usage: /todo add <task>")
                return
            self._todos.append({"text": parts[2].strip(), "done": False})
            self._append_system(f"Added TODO #{len(self._todos)}.")
            return

        if action == "done":
            if len(parts) < 3 or not parts[2].strip().isdigit():
                self._append_system("Usage: /todo done <number>")
                return
            idx = int(parts[2].strip()) - 1
            if idx < 0 or idx >= len(self._todos):
                self._append_system("TODO number out of range.")
                return
            self._todos[idx]["done"] = True
            self._append_system(f"Marked TODO #{idx + 1} done.")
            return

        self._append_system("Unknown /todo command. Use /todo, /todo add <task>, /todo done <n>.")


class ConfirmModal(ModalScreen[bool]):
    """Simple yes / no confirmation modal."""
    BINDINGS = [
        Binding("y",      "confirm(True)",  "Yes"),
        Binding("n",      "confirm(False)", "No"),
        Binding("escape", "confirm(False)", "No"),
    ]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self):
        yield Vertical(
            Label(f"[bold]{escape(self._message)}[/]", id="modal-title"),
            Label(""),
            Label("  [y]  Yes"),
            Label("  [n]  No"),
            id="modal-box"
        )

    def action_confirm(self, result: bool) -> None:
        self.dismiss(result)


class RatingModal(ModalScreen[int]):
    BINDINGS = [
        Binding("1", "rate(1)", "Again"),
        Binding("2", "rate(2)", "Hard"),
        Binding("3", "rate(3)", "Good"),
        Binding("4", "rate(4)", "Easy"),
        Binding("escape", "dismiss(0)", "Skip"),
    ]

    def __init__(self, max_r: int, attempts: int) -> None:
        super().__init__()
        self.max_r    = max_r
        self.attempts = attempts

    def compose(self):
        yield Vertical(
            Label(
                f"[white]attempt {self.attempts}[/]  —  "
                f"max: [dim]{RATING_LABELS[self.max_r]}[/]",
                id="modal-title",
            ),
            Label(""),
            *[
                Label(
                    f"  [{i}]  {RATING_LABELS[i]:<7} {RATING_DESC[i]}"
                    if i <= self.max_r else
                    f"[dim]  [{i}]  {RATING_LABELS[i]:<7} {RATING_DESC[i]}  ✕[/]"
                )
                for i in range(1, 5)
            ],
            Label(""),
            Label("  [Esc] skip without updating", id="modal-skip"),
            id="modal-box"
        )

    def action_rate(self, rating: int) -> None:
        if rating <= self.max_r:
            self.dismiss(rating)


class PaperGenerateModal(ModalScreen[dict | None]):
    """Modal to generate problems from an arXiv paper."""
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
        Binding("enter", "generate", "Generate"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._generating = False

    def compose(self):
        with Vertical(id="hint-box"):
            yield Label("[bold]Generate Problems from arXiv Paper[/bold]", id="hint-title")
            yield Label("")
            yield Label("Paste an arXiv URL or paper ID:")
            yield Input(placeholder="https://arxiv.org/abs/2402.03300", id="paper-url")
            yield Label("")
            yield Label("[dim]Number of problems:[/dim]")
            yield Input(placeholder="3", value="3", id="num-problems")
            yield Label("")
            yield Label("[dim]Language (python/julia):[/dim]")
            yield Input(placeholder="python", value="python", id="language")
            yield Label("")
            yield MarkdownWidget("*Press Enter to generate, Esc to cancel*", id="paper-status")

    def action_generate(self) -> None:
        if self._generating:
            return
        
        url_input = self.query_one("#paper-url", Input)
        url = url_input.value.strip()
        if not url:
            self.query_one("#paper-status", MarkdownWidget).update("**Please enter a paper URL**")
            return
        
        self._generating = True
        num_problems = int(self.query_one("#num-problems", Input).value or "3")
        language = self.query_one("#language", Input).value or "python"
        
        self.query_one("#paper-status", MarkdownWidget).update("*Fetching paper and generating problems...*")
        
        # Return config for the caller to handle generation
        self.dismiss({
            "url": url,
            "num_problems": num_problems,
            "language": language,
        })


class ImportProblemModal(ModalScreen[dict | None]):
    """Import problems from Exercism or LeetCode."""
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
        Binding("enter", "import_selected", "Import"),
    ]

    SOURCES = [
        ("exercism", "🟦 Exercism", "Free exercises with tests, 65+ languages"),
        ("leetcode", "🟧 LeetCode", "Classic coding problems with test cases"),
    ]

    EXERCISM_TRACK = "python"

    def __init__(self) -> None:
        super().__init__()
        self._selected_source = "exercism"
        self._items: list[dict] = []
        self._loading = False

    def compose(self):
        with Vertical(id="hint-box"):
            yield Label("[bold]Import Problems[/bold]", id="hint-title")
            yield Label("")
            yield Label("[dim]Source:[/dim]")
            yield ListView(
                *[ListItem(Label(f"{name}  [dim]{desc}[/dim]"), id=f"src-{src}") 
                  for src, name, desc in self.SOURCES],
                id="import-sources",
            )
            yield Label("")
            yield Input(placeholder="Filter (e.g., easy, python, arrays)...", id="import-filter")
            yield ListView(id="import-problems")
            yield MarkdownWidget("", id="import-status")
            yield Label("[dim]Select source → browse problems → Enter to import[/dim]", id="modal-skip")

    def on_mount(self) -> None:
        # Select first source by default
        sources = self.query_one("#import-sources", ListView)
        sources.index = 0
        self._load_source("exercism")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return
        
        item_id = event.item.id
        
        if item_id.startswith("src-"):
            # Source selected
            source = item_id[4:]
            self._selected_source = source
            self._load_source(source)
        elif item_id.startswith("prob-"):
            # Problem selected - import it
            idx = int(item_id.split("-")[1])
            if idx < len(self._items):
                self._import_problem(self._items[idx])

    def _load_source(self, source: str) -> None:
        self._loading = True
        self.query_one("#import-status", MarkdownWidget).update("*Loading...*")
        self.query_one("#import-problems", ListView).clear()
        
        def _fetch():
            items = []
            try:
                if source == "exercism":
                    from exercism import list_exercises

                    practice = list_exercises(self.EXERCISM_TRACK, "practice")
                    concept = list_exercises(self.EXERCISM_TRACK, "concept")
                    items = sorted(
                        practice + concept,
                        key=lambda item: (
                            item.get("difficulty") != "easy",
                            item.get("difficulty") == "hard",
                            item.get("type") != "concept",
                            item.get("name", ""),
                        ),
                    )
                elif source == "leetcode":
                    from leetcode import free_problems

                    items = free_problems()
            except Exception as e:
                self.app.call_from_thread(
                    self.query_one("#import-status", MarkdownWidget).update,
                    f"*Error: {e}*"
                )
                return
            
            self.app.call_from_thread(self._display_items, items)
        
        threading.Thread(target=_fetch, daemon=True).start()

    def _display_items(self, items: list[dict]) -> None:
        self._items = items
        self._loading = False
        
        list_view = self.query_one("#import-problems", ListView)
        list_view.clear()
        
        if not items:
            self.query_one("#import-status", MarkdownWidget).update("*No problems found*")
            return
        
        for i, item in enumerate(items):
            list_view.append(ListItem(Label(self._format_import_item(item)), id=f"prob-{i}"))
        
        self.query_one("#import-status", MarkdownWidget).update(f"*{len(items)} problems*")

    def _format_import_item(self, item: dict) -> str:
        name = item.get("name", item.get("title", item.get("slug", "unknown")))
        diff = item.get("difficulty", "")
        kind = item.get("type", "")
        desc = item.get("description", "")[:60]

        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(diff, "")
        label = f"{diff_icon} {name}"
        if kind:
            label += f"  [dim]({escape(kind)})[/dim]"
        if desc:
            label += f"\n  [dim]{escape(desc)}[/dim]"
        return label

    def _import_problem(self, item: dict) -> None:
        self.query_one("#import-status", MarkdownWidget).update("*Importing...*")
        
        def _do_import():
            try:
                output_dir = get_runtime().problems_dir / "imported"
                
                if self._selected_source == "exercism":
                    from exercism import fetch_and_convert, write_exercism_problem
                    slug = item.get("slug", "")
                    problem = fetch_and_convert("python", slug)
                    if problem:
                        path = write_exercism_problem(problem, output_dir)
                        self.app.call_from_thread(self._import_done, path, item)
                    else:
                        self.app.call_from_thread(
                            self.query_one("#import-status", MarkdownWidget).update,
                            "*Failed to fetch exercise*"
                        )
                
                elif self._selected_source == "leetcode":
                    from leetcode import fetch_and_convert, write_leetcode_problem
                    slug = item.get("slug", "")
                    problem = fetch_and_convert(slug)
                    if problem:
                        path = write_leetcode_problem(problem, output_dir)
                        self.app.call_from_thread(self._import_done, path, item)
                    else:
                        self.app.call_from_thread(
                            self.query_one("#import-status", MarkdownWidget).update,
                            "*Failed to fetch problem (may be premium)*"
                        )
                
            except Exception as e:
                self.app.call_from_thread(
                    self.query_one("#import-status", MarkdownWidget).update,
                    f"*Error: {e}*"
                )
        
        threading.Thread(target=_do_import, daemon=True).start()

    def _import_done(self, path: Path, item: dict) -> None:
        name = item.get("name", item.get("title", ""))
        self.query_one("#import-status", MarkdownWidget).update(
            f"**Imported: {name}**\n`{path}`"
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "import-filter":
            # Filter the current list
            query = event.value.strip().lower()
            list_view = self.query_one("#import-problems", ListView)
            list_view.clear()
            
            filtered = []
            for i, item in enumerate(self._items):
                name = item.get("name", item.get("title", "")).lower()
                desc = item.get("description", "").lower()
                diff = item.get("difficulty", "").lower()
                kind = item.get("type", "").lower()
                
                if not query or query in name or query in desc or query in diff or query in kind:
                    filtered.append((i, item))
            
            for orig_i, item in filtered:
                list_view.append(ListItem(Label(self._format_import_item(item)), id=f"prob-{orig_i}"))


class CollectionSelectModal(ModalScreen[Path]):
    """Modal to select a problem collection (folder)."""
    BINDINGS = [
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, collections: list[Path], current: Path) -> None:
        super().__init__()
        self.collections = collections
        self.current     = current

    def compose(self):
        # We'll re-use the modal-box styling but maybe make it taller via CSS in App
        yield Vertical(
            Label("[bold]Select Collection[/]", id="modal-title"),
            ListView(
                *[
                    ListItem(
                        Label(
                            f"[green]●[/] {self._display_name(c)}" if c == self.current
                            else f"  {self._display_name(c)}"
                        ),
                        id=f"col-{i}"
                    )
                    for i, c in enumerate(self.collections)
                ],
                id="collection-list"
            ),
            id="collection-box"  # new ID for styling if needed
        )

    def _display_name(self, path: Path) -> str:
        """Show a friendly name relative to the problems root."""
        if path == self.collections[0] and path.name == "problems":
            return "Main Collection"
        try:
            # Find the top-level problems dir (first collection entry)
            root = self.collections[0] if self.collections[0].name == "problems" else path.parent
            rel = path.relative_to(root)
            return str(rel) if str(rel) != "." else "Main Collection"
        except ValueError:
            return path.name

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return
        idx = int(event.item.id.split("-")[1])
        self.dismiss(self.collections[idx])
