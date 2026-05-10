"""Markdown renderers for interactive-shell streamed LLM output."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import redirect_stdout

from rich.console import Console
from rich.markdown import Markdown

NextChunk = Callable[[Iterator[str]], str | None]
_STREAMDOWN_GREEN_HUE = 0.34
_STREAMDOWN_GREEN_VALUE = 0.35


class _ChunkByteStream:
    """Expose a text chunk iterator as the byte stream Streamdown expects."""

    def __init__(
        self,
        *,
        chunks_iter: Iterator[str],
        next_chunk: NextChunk,
        on_chunk: Callable[[str], None],
    ) -> None:
        self._chunks_iter = chunks_iter
        self._next_chunk = next_chunk
        self._on_chunk = on_chunk
        self._buffer = b""
        self._closed = False

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            return b""
        if size is None or size < 0:
            parts: list[bytes] = []
            while True:
                data = self.read(8192)
                if not data:
                    break
                parts.append(data)
            return b"".join(parts)
        if size == 0:
            return b""
        while not self._buffer:
            chunk = self._next_chunk(self._chunks_iter)
            if chunk is None:
                self._closed = True
                return b""
            if not chunk:
                continue
            self._on_chunk(chunk)
            self._buffer = chunk.encode("utf-8")
        out = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return out


def console_file_supports_streamdown(console: Console) -> bool:
    """Return true when Streamdown can safely write to this console output."""

    out = console.file
    isatty = getattr(out, "isatty", None)
    if not bool(isatty and isatty()):
        return False
    fileno = getattr(out, "fileno", None)
    if fileno is None:
        return False
    try:
        fileno()
    except (OSError, ValueError, TypeError):
        return False
    return True


def render_streamdown_markdown(
    *,
    console: Console,
    chunks_iter: Iterator[str],
    next_chunk: NextChunk,
    on_chunk: Callable[[str], None],
) -> None:
    """Render Markdown incrementally with Streamdown."""

    from streamdown import Streamdown  # type: ignore[import-untyped]

    stream = _ChunkByteStream(
        chunks_iter=chunks_iter,
        next_chunk=next_chunk,
        on_chunk=on_chunk,
    )
    renderer = Streamdown()
    renderer.setup(H=_STREAMDOWN_GREEN_HUE, V=_STREAMDOWN_GREEN_VALUE)
    with redirect_stdout(console.file):
        renderer.render(stream)
        renderer.tidyup()


def render_final_markdown(console: Console, text: str) -> None:
    """Render a completed Markdown document with Rich."""

    if text:
        console.print(Markdown(text, code_theme="ansi_dark"))


__all__ = [
    "console_file_supports_streamdown",
    "render_final_markdown",
    "render_streamdown_markdown",
]
