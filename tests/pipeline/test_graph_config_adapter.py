from __future__ import annotations

import warnings

from app.pipeline.graph import build_graph


def test_build_graph_does_not_warn_about_config_annotation() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        build_graph()

    assert not any(
        "config' parameter should be typed as 'RunnableConfig'" in str(warning.message)
        for warning in captured
    )
