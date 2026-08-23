"""Text/label renderer, with automatic overlap avoidance via adjustText.

Role columns:
    x, y    required, label position
    text    required, label content
    color   optional per-label color

``adjustText`` is optional (see requirements.txt): installed, it nudges
overlapping labels apart and draws a leader line back to their original
point; missing, every label is placed at its own point exactly and may
overlap its neighbours, same as calling ``Axes.text`` directly. The
difference is only in whether labels can occupy the same pixels, not in
whether text is drawn at all - a chart with no optional dependency installed
should never come up blank because of it.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.charts.base import BaseAxisRenderer, SeriesData
from app.logs.logger import applogger

_ADJUST_TEXT_WARNED = False


def _adjust_text_function():
    """Return adjustText.adjust_text, or None with one explanation in the log.

    Memoized at module level so the missing-dependency notice is logged once
    per process rather than once per render.
    """
    global _ADJUST_TEXT_WARNED
    try:
        from adjustText import adjust_text
        return adjust_text
    except Exception:
        if not _ADJUST_TEXT_WARNED:
            _ADJUST_TEXT_WARNED = True
            applogger.info(
                "Text labels are not overlap-adjusted: the optional "
                "dependency 'adjustText' is not installed. Install it with "
                "'pip install adjustText' for automatic label spacing.",
                show_dialog=False,
                raise_error=False,
            )
        return None


class TextAxisRenderer(BaseAxisRenderer):
    """Places one text label per row, optionally spread apart by adjustText."""

    Name: str = "Text"
    Category: str = "Pairwise data"
    Description: str = "Text labels at data points, spread apart to avoid overlap."
    Link: str = "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html"

    RequiredRoles: list[str] = ["x", "y", "text"]
    OptionalRoles: list[str] = ["color"]

    Kwargs: dict[str, object] = {
        "fontsize": {
            "default": 10.0,
            "type": float,
            "min": 4.0,
            "max": 72.0,
            "group": "Appearance",
            "description": "Label text size.",
        },
        "color": {
            "default": None,
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Label text color. Overridden by the color role when present.",
        },
        "ha": {
            "default": "center",
            "type": ["left", "center", "right"],
            "group": "Appearance",
            "description": "Horizontal text alignment relative to its point.",
        },
        "va": {
            "default": "center",
            "type": ["top", "center", "bottom", "baseline"],
            "group": "Appearance",
            "description": "Vertical text alignment relative to its point.",
        },
        "show_markers": {
            "default": True,
            "type": bool,
            "group": "Appearance",
            "description": "Draw a small marker at each label's own point, so a label moved by adjustText still shows where it belongs.",
        },
        "leader_color": {
            "default": "#999999",
            "type": str,
            "kind": "color",
            "group": "Appearance",
            "description": "Leader-line color connecting an adjusted label back to its point. Only drawn when adjustText is installed and actually moves a label.",
        },
    }

    def render_axis(
        self,
        ax: Any,
        series: list[SeriesData],
        options: dict[str, Any] | None = None,
    ) -> None:
        axis_options = options or {}
        valid_series = [
            sd
            for sd in series
            if (sd.style or {}).get("visible", True)
            and self.ensure_required_roles(sd.df)
            and not sd.df.empty
        ]
        if not valid_series:
            return

        texts = []
        for layer_index, sd in enumerate(valid_series):
            merged = self._merge_options(axis_options, sd.style or {})
            texts.extend(self._place_series(ax, sd, merged, layer_index))

        if not texts:
            return

        adjust_text = _adjust_text_function()
        if adjust_text is not None:
            leader_color = str(self.opt("leader_color", axis_options) or "#999999")
            adjust_text(
                texts,
                ax=ax,
                arrowprops={"arrowstyle": "-", "color": leader_color, "lw": 0.75},
            )

        self.apply_annotations(ax, axis_options)

    def _place_series(
        self,
        ax: Any,
        sd: SeriesData,
        merged_options: dict[str, Any],
        layer_index: int,
    ) -> list[Any]:
        df = sd.df
        xs = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        labels = df["text"].astype(str).tolist()

        fallback_color = str(self.opt("color", merged_options) or "") or self.series_color(
            sd.style or {}, layer_index
        )
        colors = (
            self.color_sequence_from_values(df["color"], fallback_color=fallback_color)
            if "color" in df.columns
            else [fallback_color] * len(df)
        )

        fontsize = float(str(self.opt("fontsize", merged_options)))
        ha = str(self.opt("ha", merged_options))
        va = str(self.opt("va", merged_options))
        show_markers = bool(self.opt("show_markers", merged_options))

        texts = []
        for x, y, label, color in zip(xs, ys, labels, colors):
            if not (pd.notna(x) and pd.notna(y)):
                continue
            if show_markers:
                ax.plot(x, y, marker=".", color=color, linestyle="", markersize=3)
            texts.append(
                ax.text(x, y, label, fontsize=fontsize, color=color, ha=ha, va=va)
            )
        return texts

    def _merge_options(self, axis_options: dict[str, Any], style: dict[str, Any]) -> dict[str, Any]:
        merged = dict(axis_options or {})
        axis_kwargs = dict(merged.get("axis_kwargs", {}) or {})
        axis_kwargs.update(style.get("axis_kwargs", {}) or {})
        for key, value in style.items():
            if key != "axis_kwargs":
                merged[key] = value
        merged["axis_kwargs"] = axis_kwargs
        return merged
