"""Tests for the shared streaming renderer used by interactive-shell handlers."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from contextlib import nullcontext

import pytest
from rich.console import Console

from app.cli.interactive_shell.streaming import stream_to_console


def _strip_ansi(text: str) -> str:
    """Drop ANSI escapes so assertions check the visible output."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _tty_console() -> tuple[Console, io.StringIO]:
    """Build a Console that thinks it is a terminal for renderer selection tests."""
    buf = io.StringIO()
    return (
        Console(file=buf, force_terminal=True, color_system=None, width=80, highlight=False),
        buf,
    )


def _non_tty_console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, color_system=None, width=80), buf


def _yield_chunks(chunks: list[str]) -> Iterator[str]:
    yield from chunks


class TestNonTtyFallback:
    """On a non-terminal console the helper drains, prints, and returns full text."""

    def test_drains_stream_and_prints_without_live_artifacts(self) -> None:
        console, buf = _non_tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["Hel", "lo, ", "world"]),
        )

        output = buf.getvalue()
        assert result == "Hello, world"
        assert "assistant:" in output
        assert "Hello, world" in output
        assert "thinking" not in output

    def test_suppression_drains_silently_in_non_tty(self) -> None:
        console, buf = _non_tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(['{"actions"', ":[]}"]),
            suppress_if_starts_with="{",
        )

        assert result == '{"actions":[]}'
        output = buf.getvalue()
        assert "assistant:" not in output
        assert '{"actions"' not in output


class TestTtyRenderContract:
    """Terminal-like captures use the safe final-render path in tests."""

    def test_renders_label_and_content_as_markdown(self) -> None:
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["Run **opensre", " investigate** to start."]),
        )

        output = _strip_ansi(buf.getvalue())
        assert result == "Run **opensre investigate** to start."
        assert "assistant:" in output
        assert "**opensre" not in output
        assert "opensre investigate" in output

    def test_returns_empty_string_when_stream_is_empty(self) -> None:
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks([]),
        )

        assert result == ""
        assert "assistant:" in _strip_ansi(buf.getvalue())


class TestMidStreamError:
    """Errors inside the stream propagate while the partial buffer stays visible."""

    def test_exception_propagates_with_partial_visible(self) -> None:
        def broken_stream() -> Iterator[str]:
            yield "partial "
            yield "answer"
            raise RuntimeError("upstream 503")

        console, buf = _tty_console()

        with pytest.raises(RuntimeError, match="upstream 503"):
            stream_to_console(
                console,
                label="assistant",
                chunks=broken_stream(),
            )

        output = _strip_ansi(buf.getvalue())
        assert "partial answer" in output

    def test_single_keyboard_interrupt_is_noted_and_stream_completes(self) -> None:
        class ChunksThenSingleKbd:
            def __init__(self) -> None:
                self.i = 0
                self.raised = False

            def __iter__(self) -> Iterator[str]:
                return self

            def __next__(self) -> str:
                parts = ("partial ", "answer")
                if self.i < len(parts):
                    chunk = parts[self.i]
                    self.i += 1
                    return chunk
                if not self.raised:
                    self.raised = True
                    raise KeyboardInterrupt
                raise StopIteration

        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=iter(ChunksThenSingleKbd()),
        )

        output = _strip_ansi(buf.getvalue())
        assert "partial answer" in output
        assert "Press Ctrl+C again to stop" in output
        assert result == "partial answer"

    def test_double_keyboard_interrupt_propagates(self) -> None:
        class ChunksThenDoubleKbd:
            def __init__(self) -> None:
                self.i = 0

            def __iter__(self) -> Iterator[str]:
                return self

            def __next__(self) -> str:
                parts = ("partial ", "answer")
                if self.i < len(parts):
                    chunk = parts[self.i]
                    self.i += 1
                    return chunk
                raise KeyboardInterrupt

        console, buf = _tty_console()
        with pytest.raises(KeyboardInterrupt):
            stream_to_console(
                console,
                label="assistant",
                chunks=iter(ChunksThenDoubleKbd()),
            )

        output = _strip_ansi(buf.getvalue())
        assert "partial answer" in output
        assert "Press Ctrl+C again to stop" in output


class TestTimingFooter:
    """A small dim timing footer appears after a rendered live response."""

    def test_footer_printed_after_streamed_response(self) -> None:
        console, buf = _tty_console()
        stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["hello"]),
        )

        output = _strip_ansi(buf.getvalue())
        assert re.search(r"·\s+\d+\.\d+s", output) is not None

    def test_footer_skipped_when_stream_is_empty(self) -> None:
        console, buf = _tty_console()
        stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks([]),
        )

        output = _strip_ansi(buf.getvalue())
        assert re.search(r"·\s+\d+\.\d+s", output) is None

    def test_footer_skipped_when_response_is_suppressed(self) -> None:
        console, buf = _tty_console()
        stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(['{"actions"', ":[]}"]),
            suppress_if_starts_with="{",
        )

        output = _strip_ansi(buf.getvalue())
        assert re.search(r"·\s+\d+\.\d+s", output) is None


class TestSuppressionPeek:
    """``suppress_if_starts_with`` skips live rendering for caller-handled content."""

    def test_suppresses_and_drains_when_first_char_matches(self) -> None:
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(['{"actions"', ":[]", "}"]),
            suppress_if_starts_with="{",
        )

        assert result == '{"actions":[]}'
        output = _strip_ansi(buf.getvalue())
        assert "assistant:" not in output
        assert '{"actions"' not in output

    def test_renders_normally_when_first_char_does_not_match(self) -> None:
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["Hello, ", "world"]),
            suppress_if_starts_with="{",
        )

        assert result == "Hello, world"
        output = _strip_ansi(buf.getvalue())
        assert "assistant:" in output
        assert "Hello, world" in output

    def test_skips_leading_whitespace_before_deciding(self) -> None:
        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["  \n", '{"action"', ':"slash"}']),
            suppress_if_starts_with="{",
        )

        assert result == '  \n{"action":"slash"}'
        output = _strip_ansi(buf.getvalue())
        assert "assistant:" not in output


class TestRendererSelection:
    """The shared helper chooses one renderer path without repainting snapshots."""

    def test_force_terminal_capture_uses_one_final_markdown_render(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.cli.interactive_shell import streaming as streaming_module

        rendered: list[str] = []

        def fake_render_final(_console: Console, text: str) -> None:
            rendered.append(text)

        monkeypatch.setattr(streaming_module, "render_final_markdown", fake_render_final)

        console, _ = _tty_console()
        chunks = (f"chunk{i} " for i in range(100))
        result = stream_to_console(console, label="assistant", chunks=chunks)

        assert "chunk0" in result
        assert "chunk99" in result
        assert rendered == [result]

    def test_streamdown_path_receives_peeked_and_remaining_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.cli.interactive_shell import streaming as streaming_module

        seen: list[str] = []

        def fake_streamdown(
            *,
            console: Console,
            chunks_iter: Iterator[str],
            next_chunk,
            on_chunk,
        ) -> None:
            while True:
                chunk = next_chunk(chunks_iter)
                if chunk is None:
                    break
                on_chunk(chunk)
                seen.append(chunk)
            console.print("streamdown-rendered")

        monkeypatch.setattr(streaming_module, "console_file_supports_streamdown", lambda _: True)
        monkeypatch.setattr(streaming_module, "patch_stdout", lambda **_kwargs: nullcontext())
        monkeypatch.setattr(streaming_module, "render_streamdown_markdown", fake_streamdown)

        console, buf = _tty_console()
        result = stream_to_console(
            console,
            label="assistant",
            chunks=_yield_chunks(["Hello", " **world**"]),
            suppress_if_starts_with="{",
        )

        assert result == "Hello **world**"
        assert seen == ["Hello", " **world**"]
        assert "streamdown-rendered" in _strip_ansi(buf.getvalue())
