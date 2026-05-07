"""Helpers for the editable pipeline configuration shown in the UI.

The Streamlit page owns the widget state, but the rules that keep the editor
synced with the generated config are pure and easy to test here.
"""

from __future__ import annotations

import json
from typing import Any


def config_to_text(config: dict[str, Any]) -> str:
    """Serialize a pipeline configuration to a stable, human-editable JSON string."""
    return json.dumps(config, indent=2)


def parse_config_text(raw_text: str) -> dict[str, Any]:
    """Parse a JSON config string and require the top-level value to be a mapping."""
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("La configurazione deve essere un oggetto JSON.")
    return parsed


def sync_editor_text(
    current_text: str | None,
    baseline_text: str | None,
    generated_text: str,
) -> tuple[str, str]:
    """Refresh the editor until the user diverges from the last generated snapshot."""
    if current_text is None:
        return generated_text, generated_text

    if baseline_text is not None and current_text == baseline_text:
        return generated_text, generated_text

    return current_text, generated_text


def should_refresh_editor_widget(
    widget_text: str | None,
    previous_baseline_text: str | None,
) -> bool:
    """
    Return True when the visible editor still mirrors the previous generated config.

    Streamlit widgets keep their own state across reruns. The decision must be
    based on the previous baseline, otherwise the text area can keep showing an
    old generated config after composer controls changed.
    """
    return widget_text is None or widget_text == previous_baseline_text
