"""CLI commands for reading and writing local OpenSRE config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from app.constants import OPENSRE_HOME_DIR

_SUPPORTED_LAYOUTS = {"classic", "pinned"}
_SUPPORTED_KEYS = ("interactive.enabled", "interactive.layout")


def _config_path() -> Path:
    return OPENSRE_HOME_DIR / "config.yml"


def _load_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    if not isinstance(data, dict):
        return {}
    return data


def _save_config(data: dict[str, Any]) -> None:
    import yaml  # type: ignore[import-untyped]

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _parse_bool(raw_value: str) -> bool:
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise click.UsageError(
        "Invalid value for interactive.enabled. "
        "Use one of: true/false, 1/0, yes/no, on/off."
    )


def _coerce_value(key: str, raw_value: str) -> bool | str:
    if key == "interactive.enabled":
        return _parse_bool(raw_value)
    if key == "interactive.layout":
        layout = raw_value.strip().lower()
        if layout not in _SUPPORTED_LAYOUTS:
            raise click.UsageError("Invalid value for interactive.layout. Use 'classic' or 'pinned'.")
        return layout
    raise click.UsageError(f"Unknown config key '{key}'. Supported keys: {', '.join(_SUPPORTED_KEYS)}")


def _set_nested_key(data: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    head, tail = dotted_key.split(".", 1)
    node = data.get(head)
    if not isinstance(node, dict):
        node = {}
    node[tail] = value
    data[head] = node
    return data


@click.group(name="config")
def config_command() -> None:
    """Inspect and update local CLI config."""


@config_command.command(name="show")
def config_show() -> None:
    """Show resolved interactive config values."""
    from app.cli.interactive_shell.config import ReplConfig
    from app.cli.support.context import is_json_output

    resolved = ReplConfig.load()
    payload = {"interactive": {"enabled": resolved.enabled, "layout": resolved.layout}}

    if is_json_output():
        click.echo(json.dumps(payload))
        return

    import yaml  # type: ignore[import-untyped]

    click.echo(yaml.safe_dump(payload, sort_keys=False).rstrip())


@config_command.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set one local config key in ~/.opensre/config.yml."""
    key = key.strip()
    coerced = _coerce_value(key, value)
    data = _load_config()
    updated = _set_nested_key(data, key, coerced)
    _save_config(updated)
    click.echo(f"✓ Set {key} = {coerced}")
