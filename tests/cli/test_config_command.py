from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from app.cli.__main__ import cli


def _patch_config_home(monkeypatch, tmp_path: Path) -> Path:
    opensre_home = tmp_path / ".opensre"
    monkeypatch.setattr("app.constants.OPENSRE_HOME_DIR", opensre_home)
    monkeypatch.setattr("app.cli.commands.config.OPENSRE_HOME_DIR", opensre_home)
    return opensre_home


def test_config_show_outputs_interactive_block(monkeypatch, tmp_path: Path) -> None:
    _patch_config_home(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "show"])

    assert result.exit_code == 0
    assert "interactive:" in result.output
    assert "enabled:" in result.output
    assert "layout:" in result.output


def test_config_set_round_trips_layout(monkeypatch, tmp_path: Path) -> None:
    opensre_home = _patch_config_home(monkeypatch, tmp_path)
    runner = CliRunner()

    set_result = runner.invoke(cli, ["config", "set", "interactive.layout", "pinned"])
    assert set_result.exit_code == 0
    assert "interactive.layout = pinned" in set_result.output

    config_path = opensre_home / "config.yml"
    assert config_path.exists()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["interactive"]["layout"] == "pinned"

    show_result = runner.invoke(cli, ["config", "show"])
    assert show_result.exit_code == 0
    assert "layout: pinned" in show_result.output


def test_config_set_unknown_key_returns_helpful_error(monkeypatch, tmp_path: Path) -> None:
    _patch_config_home(monkeypatch, tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["config", "set", "foo.bar", "value"])

    assert result.exit_code != 0
    assert "Unknown config key 'foo.bar'" in result.output
    assert "interactive.enabled" in result.output
    assert "interactive.layout" in result.output
