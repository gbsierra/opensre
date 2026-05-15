"""Help renderers for the interactive shell slash-command catalog."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from app.cli.interactive_shell.command_registry.types import SlashCommand
from app.cli.interactive_shell.ui.choice_menu import (
    erase_menu_lines,
    menu_columns,
    read_menu_action,
    write_menu_line,
)
from app.cli.interactive_shell.ui.rendering import repl_table
from app.cli.interactive_shell.ui.theme import (
    ANSI_RESET,
    BOLD_BRAND,
    BOLD_BRAND_ANSI,
    DIM,
    DIM_COUNTER_ANSI,
    HIGHLIGHT,
    HIGHLIGHT_ANSI,
    MENU_SELECTION_ROW_ANSI,
    PROMPT_ACCENT_ANSI,
)

HelpSection = tuple[str, Sequence[SlashCommand]]
_HELP_VIEW_ROWS = 14
_HELP_HINT = "↑↓/j/k  Enter toggle details  Esc/q close"


@dataclass(frozen=True)
class HelpRow:
    section: str | None = None
    command: SlashCommand | None = None

    @property
    def selectable(self) -> bool:
        return self.command is not None


@dataclass(frozen=True)
class HelpDisplayRow:
    source_index: int | None = None
    section: str | None = None
    command: SlashCommand | None = None
    detail: str | None = None


def render_help_index(console: Console, sections: Sequence[HelpSection]) -> None:
    """Render the compact non-interactive help index."""
    table = repl_table(title="Slash commands", title_style=BOLD_BRAND, show_header=False)
    table.add_column("command", no_wrap=True, min_width=18)
    table.add_column("description", style=DIM)

    for section_name, commands in sections:
        if not commands:
            continue
        table.add_row(f"[{BOLD_BRAND}]{escape(section_name)}[/]", "")
        for index, command in enumerate(commands):
            table.add_row(
                f"  [{HIGHLIGHT}]{escape(command.name)}[/]",
                escape(command.description),
                end_section=(index == len(commands) - 1),
            )

    console.print(table)
    console.print(f"[{DIM}]Use[/] [bold]/help <command>[/bold] [{DIM}]for usage.[/]")


def render_section_detail(
    console: Console,
    section_name: str,
    commands: Sequence[SlashCommand],
) -> None:
    """Render one category using the same compact description-only style."""
    table = repl_table(title=f"{section_name} commands", title_style=BOLD_BRAND, show_header=False)
    table.add_column("command", no_wrap=True, min_width=18)
    table.add_column("description", style=DIM)
    for command in commands:
        table.add_row(f"[{HIGHLIGHT}]{escape(command.name)}[/]", escape(command.description))
    console.print(table)
    console.print(f"[{DIM}]Use[/] [bold]/help <command>[/bold] [{DIM}]for usage.[/]")


def render_command_detail(console: Console, command: SlashCommand) -> None:
    """Render detailed help for one slash command."""
    table = Table(title=command.name, title_style=BOLD_BRAND, show_header=False, box=None)
    table.add_column("label", style="bold", no_wrap=True)
    table.add_column("value")
    table.add_row("description", escape(command.description))

    if command.usage:
        table.add_row("usage", "\n".join(escape(item) for item in command.usage))
    if command.examples:
        table.add_row("examples", "\n".join(escape(item) for item in command.examples))
    if command.notes:
        table.add_row("notes", "\n".join(escape(item) for item in command.notes))

    console.print(table)


def has_help_details(command: SlashCommand) -> bool:
    """True when a command has expandable usage, examples, or notes."""
    return bool(command.usage or command.examples or command.notes)


def _flatten_help_rows(sections: Sequence[HelpSection]) -> list[HelpRow]:
    rows: list[HelpRow] = []
    for section_name, commands in sections:
        if not commands:
            continue
        rows.append(HelpRow(section=section_name))
        rows.extend(HelpRow(command=command) for command in commands)
    return rows


def _first_selectable_index(rows: Sequence[HelpRow]) -> int | None:
    for index, row in enumerate(rows):
        if row.selectable:
            return index
    return None


def _next_selectable_index(rows: Sequence[HelpRow], current: int, delta: int) -> int:
    if not rows:
        return current
    index = current
    for _ in range(len(rows)):
        index = (index + delta) % len(rows)
        if rows[index].selectable:
            return index
    return current


def _expanded_detail_lines(command: SlashCommand) -> list[str]:
    lines: list[str] = []
    if command.usage:
        lines.append("usage:")
        lines.extend(f"  {item}" for item in command.usage)
    if command.examples:
        lines.append("examples:")
        lines.extend(f"  {item}" for item in command.examples)
    if command.notes:
        lines.append("notes:")
        lines.extend(f"  {item}" for item in command.notes)
    return lines


def _display_rows(rows: Sequence[HelpRow], expanded: int | None) -> list[HelpDisplayRow]:
    display: list[HelpDisplayRow] = []
    for index, row in enumerate(rows):
        display.append(
            HelpDisplayRow(
                source_index=index,
                section=row.section,
                command=row.command,
            )
        )
        if expanded == index and row.command is not None:
            display.extend(
                HelpDisplayRow(detail=line) for line in _expanded_detail_lines(row.command)
            )
    return display


def _display_index_for_source(display_rows: Sequence[HelpDisplayRow], source_index: int) -> int:
    for index, row in enumerate(display_rows):
        if row.source_index == source_index:
            return index
    return 0


def _detail_end_index(rows: Sequence[HelpDisplayRow], selected: int) -> int:
    end = selected + 1
    while end < len(rows) and rows[end].detail is not None:
        end += 1
    return end


def _expanded_viewport_height(
    rows: Sequence[HelpDisplayRow],
    selected: int,
    base_height: int,
) -> int:
    detail_end = _detail_end_index(rows, selected)
    expanded_block_height = detail_end - selected
    if expanded_block_height <= 1:
        return base_height
    return max(base_height, expanded_block_height + 2)


def _viewport_bounds(
    rows: Sequence[HelpDisplayRow],
    selected: int,
    height: int,
) -> tuple[int, int]:
    if len(rows) <= height:
        return 0, len(rows)

    before = max(0, height // 3)
    start = max(0, selected - before)
    end = start + height
    if end > len(rows):
        end = len(rows)
        start = max(0, end - height)

    detail_end = _detail_end_index(rows, selected)
    if detail_end > selected + 1 and detail_end - selected <= height:
        if selected < start:
            start = selected
            end = min(len(rows), start + height)
        if detail_end > end:
            end = detail_end
            start = max(0, end - height)

    # Keep the selected command's category header visible when it fits in the viewport.
    section_start = selected
    while section_start > 0 and rows[section_start].section is None:
        section_start -= 1
    if rows[section_start].section is not None and section_start < start:
        detail_end = _detail_end_index(rows, selected)
        distance = detail_end - section_start
        if distance <= height:
            start = section_start
            end = min(len(rows), start + height)
    return start, end


def _visible_width(text: str) -> int:
    return len(text)


def _clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _visible_width(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def _pad(text: str, width: int) -> str:
    clipped = _clip(text, width)
    return clipped + (" " * max(0, width - _visible_width(clipped)))


def _render_command_row(
    command: SlashCommand,
    *,
    selected: bool,
    expanded: bool,
    width: int,
) -> str:
    marker = ">" if selected else " "
    affordance = "▾" if expanded else "▸" if has_help_details(command) else " "
    name_width = min(18, max(12, width // 3))
    desc_width = max(0, width - name_width - 7)
    plain = (
        f" {marker} {affordance} {_pad(command.name, name_width)} "
        f"{_clip(command.description, desc_width)}"
    )
    padded = _pad(plain, width)
    if selected:
        return f"{MENU_SELECTION_ROW_ANSI}{padded}{ANSI_RESET}"
    return (
        f"{DIM_COUNTER_ANSI}   {affordance} {ANSI_RESET}"
        f"{HIGHLIGHT_ANSI}{_pad(command.name, name_width)}{ANSI_RESET} "
        f"{DIM_COUNTER_ANSI}{_clip(command.description, desc_width)}{ANSI_RESET}"
    )


def _render_help_row(row: HelpRow, *, selected: bool, expanded: bool, width: int) -> str:
    if row.section is not None:
        return f"{BOLD_BRAND_ANSI}{_pad(row.section, width)}{ANSI_RESET}"
    if row.command is None:
        return ""
    return _render_command_row(row.command, selected=selected, expanded=expanded, width=width)


def _render_display_row(
    row: HelpDisplayRow,
    *,
    selected: bool,
    expanded: bool,
    width: int,
) -> str:
    if row.detail is not None:
        return f"{DIM_COUNTER_ANSI}    {_clip(row.detail, max(0, width - 4))}{ANSI_RESET}"
    return _render_help_row(
        HelpRow(section=row.section, command=row.command),
        selected=selected,
        expanded=expanded,
        width=width,
    )


def _help_menu_height(viewport_height: int) -> int:
    # leading blank, title, counter, rule, rows, blank, hint
    return 5 + viewport_height + 1


def _draw_help_menu(
    rows: Sequence[HelpRow],
    *,
    selected: int,
    expanded: int | None,
    erase_lines: int,
    viewport_height: int = _HELP_VIEW_ROWS,
) -> int:
    width = menu_columns()
    display = _display_rows(rows, expanded)
    display_selected = _display_index_for_source(display, selected)
    effective_viewport_height = _expanded_viewport_height(
        display,
        display_selected,
        viewport_height,
    )
    start, end = _viewport_bounds(display, display_selected, effective_viewport_height)
    visible = display[start:end]
    height = _help_menu_height(effective_viewport_height)
    if erase_lines:
        erase_menu_lines(erase_lines)

    selected_count = sum(1 for row in rows[: selected + 1] if row.selectable)
    total_count = sum(1 for row in rows if row.selectable)

    write_menu_line()
    write_menu_line(f"{PROMPT_ACCENT_ANSI}Slash commands{ANSI_RESET}")
    write_menu_line(f"{DIM_COUNTER_ANSI}{selected_count}/{total_count}{ANSI_RESET}")
    write_menu_line(f"{DIM_COUNTER_ANSI}{'─' * width}{ANSI_RESET}")
    for offset, row in enumerate(visible, start=start):
        write_menu_line(
            _render_display_row(
                row,
                selected=(offset == display_selected),
                expanded=(row.source_index == expanded),
                width=width,
            )
        )
    for _ in range(max(0, effective_viewport_height - len(visible))):
        write_menu_line()
    write_menu_line()
    write_menu_line(f"{DIM_COUNTER_ANSI}{_HELP_HINT}{ANSI_RESET}")
    sys.stdout.flush()
    return height


def choose_help_command(sections: Sequence[HelpSection]) -> SlashCommand | None:
    """Let a TTY user browse command details from a grouped viewport."""
    rows = _flatten_help_rows(sections)
    selected = _first_selectable_index(rows)
    if selected is None:
        return None

    erase_lines = 0
    expanded: int | None = None
    while True:
        erase_lines = _draw_help_menu(
            rows,
            selected=selected,
            expanded=expanded,
            erase_lines=erase_lines,
        )
        action = read_menu_action()
        if action == "enter":
            command = rows[selected].command
            if command is not None and has_help_details(command):
                expanded = None if expanded == selected else selected
            continue
        if action in ("cancel", "eof"):
            erase_menu_lines(erase_lines)
            return None
        if action == "ignore":
            continue
        if action == "up":
            selected = _next_selectable_index(rows, selected, -1)
            expanded = None
        elif action == "down":
            selected = _next_selectable_index(rows, selected, 1)
            expanded = None


__all__ = [
    "HelpSection",
    "HelpRow",
    "choose_help_command",
    "render_command_detail",
    "render_help_index",
    "render_section_detail",
]
