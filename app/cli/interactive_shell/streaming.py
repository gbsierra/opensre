"""Live token streaming for interactive-shell LLM responses."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from itertools import chain

from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from app.cli.interactive_shell.streaming_markdown import (
    console_file_supports_streamdown,
    render_final_markdown,
    render_streamdown_markdown,
)
from app.cli.interactive_shell.theme import BOLD_BRAND, DIM, HIGHLIGHT, MARKDOWN_THEME
from app.cli.support.prompt_support import CTRL_C_DOUBLE_PRESS_WINDOW_S

if sys.platform == "win32":
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
else:

    class NoConsoleScreenBufferError(Exception):
        """Only the Windows prompt_toolkit stack raises this concrete type."""


_STREAM_CANCEL_HINT = "Press Ctrl+C again to stop"
_SPINNER_NAME = "dots12"
_SPINNER_COLOR = HIGHLIGHT
_SPINNER_LABEL = "thinking"
_SPINNER_REFRESH_PER_SECOND = 10

STREAM_LABEL_ASSISTANT = "assistant"
STREAM_LABEL_ANSWER = "answer"


def _console_file_is_a_tty(console: Console) -> bool:
    """True only when Rich is writing to a real TTY, not StringIO capture."""

    out = console.file
    isatty = getattr(out, "isatty", None)
    return bool(isatty and isatty())


def _build_waiting_spinner() -> Spinner:
    return Spinner(
        _SPINNER_NAME,
        text=Text(f"{_SPINNER_LABEL}…", style=f"bold {_SPINNER_COLOR}"),
        style=f"bold {_SPINNER_COLOR}",
    )


def stream_to_console(
    console: Console,
    *,
    label: str,
    chunks: Iterator[str],
    suppress_if_starts_with: str | None = None,
) -> str:
    """Render a streaming LLM response live and return the accumulated text.

    Real terminal output uses a streaming Markdown renderer. Captured output,
    pipes, and test StringIO consoles drain the stream and render Markdown
    once at the end so they avoid terminal-control artifacts.

    ``suppress_if_starts_with`` allows callers to skip live rendering when the
    initial non-whitespace token indicates machine-readable payloads, such as
    JSON action plans.
    """
    if not console.is_terminal:
        text = "".join(chunks)
        if suppress_if_starts_with is not None and text.lstrip().startswith(
            suppress_if_starts_with
        ):
            return text
        if text:
            console.print()
            console.print(f"[{BOLD_BRAND}]{label}:[/]")
            with console.use_theme(MARKDOWN_THEME):
                render_final_markdown(console, text)
            console.print()
        return text

    chunks_iter = iter(chunks)
    peeked: list[str] = []
    first_interrupt_at: float | None = None

    def _note_stream_interrupt() -> None:
        nonlocal first_interrupt_at
        now = time.monotonic()
        if (
            first_interrupt_at is not None
            and now - first_interrupt_at <= CTRL_C_DOUBLE_PRESS_WINDOW_S
        ):
            first_interrupt_at = None
            raise KeyboardInterrupt
        first_interrupt_at = now
        console.print(f"[{DIM}]{_STREAM_CANCEL_HINT}[/]")

    def _next_chunk(it: Iterator[str]) -> str | None:
        while True:
            try:
                return next(it)
            except StopIteration:
                return None
            except KeyboardInterrupt:
                _note_stream_interrupt()

    def _read_initial_response() -> str | None:
        if suppress_if_starts_with is None:
            chunk = _next_chunk(chunks_iter)
            if chunk is not None:
                peeked.append(chunk)
            return None

        while True:
            chunk = _next_chunk(chunks_iter)
            if chunk is None:
                return None
            peeked.append(chunk)
            stripped = "".join(peeked).lstrip()
            if not stripped:
                continue
            if stripped.startswith(suppress_if_starts_with):
                drained: list[str] = []
                while True:
                    rest = _next_chunk(chunks_iter)
                    if rest is None:
                        break
                    drained.append(rest)
                return "".join(peeked) + "".join(drained)
            return None

    suppressed_text: str | None = None
    if _console_file_is_a_tty(console):
        try:
            with (
                patch_stdout(raw=True),
                Live(
                    _build_waiting_spinner(),
                    console=console,
                    refresh_per_second=_SPINNER_REFRESH_PER_SECOND,
                    transient=True,
                ),
            ):
                suppressed_text = _read_initial_response()
        except NoConsoleScreenBufferError:
            if not peeked:
                suppressed_text = _read_initial_response()
    else:
        suppressed_text = _read_initial_response()
    if suppressed_text is not None:
        return suppressed_text

    render_chunks_iter = chain(peeked, chunks_iter)
    buffer: list[str] = []

    def _record_chunk(chunk: str) -> None:
        buffer.append(chunk)

    def _drain_to_buffer() -> None:
        while True:
            chunk = _next_chunk(render_chunks_iter)
            if chunk is None:
                break
            if chunk:
                buffer.append(chunk)

    def _drain_and_render_final() -> None:
        try:
            _drain_to_buffer()
        finally:
            render_final_markdown(console, "".join(buffer))

    console.print()
    console.print(f"[{BOLD_BRAND}]{label}:[/]")

    started = time.monotonic()
    try:
        with console.use_theme(MARKDOWN_THEME):
            if console_file_supports_streamdown(console):
                try:
                    with patch_stdout(raw=True):
                        render_streamdown_markdown(
                            console=console,
                            chunks_iter=render_chunks_iter,
                            next_chunk=_next_chunk,
                            on_chunk=_record_chunk,
                        )
                except NoConsoleScreenBufferError:
                    _drain_and_render_final()
            else:
                _drain_and_render_final()
        if buffer:
            console.print(f"[{DIM}]· {time.monotonic() - started:.1f}s[/]")
    finally:
        console.print()

    return "".join(buffer)


__all__ = ["STREAM_LABEL_ANSWER", "STREAM_LABEL_ASSISTANT", "stream_to_console"]
