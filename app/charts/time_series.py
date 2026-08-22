"""Time-series renderer for numeric or timestamp x axes.

The x column is inspected rather than assumed: a real timestamp column gets a
date locator and formatter, a numeric column stays numeric.  Optional gap
insertion breaks the line where consecutive samples are further apart than a
configurable threshold, so a data outage reads as a gap instead of a straight
line through it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from app.charts.base import ERROR_BAR_KWARGS, BaseAxisRenderer, SeriesData
from app.logs.logger import applogger
from app.utils.coercion import coerce_axis


class TimeSeriesAxisRenderer(BaseAxisRenderer):
    """Render time series data with optional raw and rolling-average lines.

    Rules in this version:
    - If marker is set, markers are shown.
    - If line style is set, the line is shown.
    - If marker is empty and line style is enabled, only the line is drawn.
    - If both marker and line style are disabled, nothing is drawn.

    Per-series style fields supported:
        - label
        - visible
        - show_in_legend
        - sort_x
        - color
        - marker
        - linestyle
        - alpha
        - zorder
        - linewidth
        - rolling_linestyle
    """

    RequiredRoles = ["x", "y"]
    OptionalRoles = [
        "color",
        # Error bands: one column for a symmetric half-width, or a low/high
        # pair for an asymmetric interval.
        "xerr",
        "xerr_low",
        "xerr_high",
        "yerr",
        "yerr_low",
        "yerr_high",
    ]

    Description = "Time series"
    Name = "Time Series"
    Category: str = "Pairwise data"

    Kwargs: dict[str, object] = {
        "alpha": {
            "default": 0.8,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": "Line opacity from 0.0 (transparent) to 1.0 (opaque).",
        },
        "norm": {
            "default": None,
            "type": ["linear", "log", "symlog", "logit"],
            "description": (
                "Normalization for numeric color data before colormap mapping."
            ),
        },
        "vmin": {
            "default": None,
            "type": float,
            "min": -1_000_000_000.0,
            "max": 1_000_000_000.0,
            "description": (
                "Lower bound for colormap normalization when numeric color data is used."
            ),
        },
        "vmax": {
            "default": None,
            "type": float,
            "min": -1_000_000_000.0,
            "max": 1_000_000_000.0,
            "description": (
                "Upper bound for colormap normalization when numeric color data is used."
            ),
        },
        "plotnonfinite": {
            "default": False,
            "type": bool,
            "description": (
                "Draw points whose color values are NaN, inf, or -inf "
                "using the colormap bad color."
            ),
        },
        "label": {
            "default": None,
            "type": str,
            "description": "Legend label override. If empty, the series name is used.",
        },
        "visible": {
            "default": True,
            "type": bool,
            "description": "Whether the plotted line is visible.",
        },
        "zorder": {
            "default": None,
            "type": float,
            "min": -1000.0,
            "max": 1000.0,
            "description": "Drawing order; higher values are drawn on top.",
        },
        "picker": {
            "default": None,
            "type": float,
            "min": 0.0,
            "max": 1000.0,
            "description": "Pick tolerance in points. Leave empty to disable picking.",
        },
        "rasterized": {
            "default": False,
            "type": bool,
            "description": "Rasterize the line during vector export.",
        },
        "rolling_window": {
            "default": 15,
            "type": int,
            "min": 1,
            "max": 1000,
            "description": (
                "Window size for rolling average. Higher values produce smoother lines."
            ),
        },
        "gap_threshold": {
            "default": "2h",
            "type": str,
            "description": (
                "Time gap threshold for inserting NaN gaps in the line. "
                'Examples: "30min", "2h", "1D".'
            ),
        },
        "show_raw": {
            "default": True,
            "type": bool,
            "description": "Whether to show the raw data line.",
        },
        "show_rolling": {
            "default": True,
            "type": bool,
            "description": "Whether to show the rolling average line.",
        },
        "linewidth": {
            "default": 1.6,
            "type": float,
            "min": 0.1,
            "max": 10.0,
            "description": "Line width for the raw data line.",
        },
        "rolling_linestyle": {
            "default": "--",
            "type": str,
            "description": 'Line style for the rolling average line, e.g. "--".',
        },
        **ERROR_BAR_KWARGS,
    }

    def render_axis(self, ax, series: list[SeriesData], options: dict) -> None:
        """Render one or more time series on the given axis."""
        base_kwargs = self.get_kwargs(options)

        raw_plot_kwargs = {
            key: value
            for key, value in base_kwargs.items()
            if key
            not in {
                "show_raw",
                "show_rolling",
                "rolling_window",
                "gap_threshold",
                "rolling_linestyle",
                "norm",
                "vmin",
                "vmax",
                "plotnonfinite",
                # ax.plot does not take errorbar keywords; they go to the
                # separate ax.errorbar call below.
                *ERROR_BAR_KWARGS,
            }
        }

        show_raw = bool(base_kwargs.get("show_raw", True))
        show_rolling = bool(base_kwargs.get("show_rolling", True))
        rolling_window = int(base_kwargs.get("rolling_window", 15))
        gap_threshold = str(base_kwargs.get("gap_threshold", "2h"))
        default_rolling_linestyle = str(
            base_kwargs.get("rolling_linestyle", "--")
        )

        legend_handles_found = False
        any_temporal_axis = False

        for series_index, sd in enumerate(series):
            style = dict(sd.style or {})
            if not bool(style.get("visible", True)) or not self.ensure_required_roles(sd.df):
                continue

            x_axis, x_is_temporal = self._coerce_x_axis(sd.df["x"])
            y_num = pd.to_numeric(sd.df["y"], errors="coerce")

            mask = x_axis.notna() & y_num.notna()
            x_axis = x_axis[mask]
            y_num = y_num[mask]

            if x_axis.empty or y_num.empty:
                applogger.warning(
                    "Series '%s' skipped: no valid datetime/numeric data available.",
                    sd.name,
                )
                continue

            any_temporal_axis = any_temporal_axis or x_is_temporal
            frame = pd.DataFrame({"x": x_axis, "y": y_num})
            color_values_for_points = None
            color_is_discrete = False
            if "color" in sd.df.columns:
                color_values_for_points = sd.df.loc[mask, "color"]
                color_values_for_points = color_values_for_points.reindex(frame.index)
                frame["color"] = color_values_for_points
                color_is_discrete = self.is_discrete_integer_color(color_values_for_points)

            # Error bars use the pre-gap, pre-sort arrays so they stay aligned
            # with their own points; _insert_gaps adds NaN rows that have no
            # error value of their own.
            if self.has_error_roles(sd.df):
                self._draw_error_bars(
                    ax=ax,
                    sd=sd,
                    mask=mask,
                    x_values=x_axis.to_numpy(),
                    y_values=y_num.to_numpy(dtype=float),
                    style=style,
                    base_kwargs=base_kwargs,
                    series_index=series_index,
                )

            # Optional sort by X ascending.
            if bool(style.get("sort_x", False)):
                frame = frame.sort_values("x", kind="stable")

            frame = self._insert_gaps(frame, gap_threshold, temporal=x_is_temporal)

            show_in_legend = bool(style.get("show_in_legend", True))
            series_label = str(style.get("label", "") or "").strip()
            if not series_label:
                axis_label = base_kwargs.get("label")
                if isinstance(axis_label, str) and axis_label.strip():
                    series_label = axis_label.strip()
                else:
                    series_label = sd.name.strip()

            marker_style = str(style.get("marker", "") or "").strip()
            line_style = str(style.get("linestyle", "") or "").strip()

            if not marker_style:
                marker_option = options.get("marker")
                if isinstance(marker_option, str):
                    marker_style = marker_option.strip()

            if not line_style:
                line_option = options.get("linestyle")
                if isinstance(line_option, str):
                    line_style = line_option.strip()

            has_marker = marker_style != ""
            has_line = line_style.lower() != ""

            # Nothing to draw for this series.
            if not has_marker and not has_line and not show_rolling:
                '''applogger.warning(
                    "Series '%s': neither marker nor line style is enabled.",
                    sd.name,
                )'''
                marker_style="."
                line_style=""
                has_marker=True

            flat_color = self._series_line_color(sd.df, style, series_index)

            common_line_kwargs: dict[str, Any] = dict(raw_plot_kwargs)

            if has_line:
                common_line_kwargs["linestyle"] = line_style

            if has_marker:
                common_line_kwargs["marker"] = marker_style
            else:
                common_line_kwargs["marker"] = "None"

            if flat_color:
                common_line_kwargs["color"] = flat_color

            if "alpha" in style and style["alpha"] not in (None, ""):
                common_line_kwargs["alpha"] = style["alpha"]

            if "zorder" in style and style["zorder"] not in (None, ""):
                common_line_kwargs["zorder"] = style["zorder"]

            if "visible" in style:
                common_line_kwargs["visible"] = bool(style["visible"])

            linewidth_value = style.get("linewidth", style.get("line_width"))
            if linewidth_value not in (None, ""):
                common_line_kwargs["linewidth"] = linewidth_value

            # Raw line/marker rendering:
            # - markers only if marker exists and no line
            # - line only if line exists and no marker
            # - both if both exist
            if show_raw and (has_line or has_marker):
                raw_kwargs = dict(common_line_kwargs)

                if show_in_legend and series_label:
                    raw_kwargs["label"] = series_label
                else:
                    raw_kwargs["label"] = "_nolegend_"

                if "color" in frame.columns:
                    # Cluster/color role: ax.plot can only use one color for the
                    # whole artist, so draw same-color contiguous line segments
                    # and color markers with scatter. This mirrors scatter-axis
                    # behavior for discrete ClusterId values.
                    if has_line:
                        line_kwargs = dict(raw_kwargs)
                        line_kwargs["marker"] = "None"
                        line_kwargs.pop("color", None)
                        legend_handles_found = (
                            self._draw_color_segments(
                                ax=ax,
                                frame=frame,
                                line_kwargs=line_kwargs,
                                fallback_color=flat_color,
                                show_legend=show_in_legend,
                                label=series_label,
                            )
                            or legend_handles_found
                        )

                    if has_marker:
                        marker_kwargs = dict(raw_kwargs)
                        marker_kwargs.pop("linestyle", None)
                        marker_kwargs.pop("color", None)
                        marker_kwargs.pop("label", None)
                        marker_kwargs.pop("marker", None)
                        marker_kwargs["marker"] = marker_style
                        marker_kwargs["label"] = "_nolegend_" if has_line else raw_kwargs.get("label", "_nolegend_")
                        self._draw_colored_markers(
                            ax=ax,
                            frame=frame,
                            marker_kwargs=marker_kwargs,
                            fallback_color=flat_color,
                            color_is_discrete=color_is_discrete,
                        )
                        if (
                            not has_line
                            and show_in_legend
                            and series_label
                        ):
                            legend_handles_found = True
                else:
                    # If only marker is enabled, use no connecting line.
                    if has_marker and not has_line:
                        raw_kwargs["linestyle"] = "None"

                    raw_lines = ax.plot(
                        frame["x"].to_numpy(),
                        frame["y"].to_numpy(dtype=float),
                        **raw_kwargs,
                    )
                    if raw_lines:
                        raw_artist = raw_lines[0]
                        label_value = raw_artist.get_label()
                        if (
                            isinstance(label_value, str)
                            and label_value
                            and not label_value.startswith("_")
                            and raw_artist.get_visible()
                        ):
                            legend_handles_found = True

            if show_rolling:
                rolling_kwargs = dict(common_line_kwargs)
                rolling_linestyle = str(
                    style.get(
                        "rolling_linestyle",
                        default_rolling_linestyle,
                    )
                    or default_rolling_linestyle
                )
                rolling_kwargs["linestyle"] = rolling_linestyle

                # Rolling line should not show markers unless explicitly desired.
                if not has_marker:
                    rolling_kwargs["marker"] = "None"

                if show_in_legend and series_label:
                    if show_raw and (has_line or has_marker):
                        rolling_kwargs["label"] = f"{series_label} (rolling)"
                    else:
                        rolling_kwargs["label"] = series_label
                else:
                    rolling_kwargs["label"] = "_nolegend_"

                roll = (
                    frame.set_index("x")["y"]
                    .rolling(
                        rolling_window,
                        min_periods=max(2, rolling_window // 3),
                    )
                    .mean()
                )

                rolling_lines = ax.plot(
                    roll.index.to_numpy(),
                    roll.to_numpy(dtype=float),
                    **rolling_kwargs,
                )
                if rolling_lines:
                    rolling_artist = rolling_lines[0]
                    label_value = rolling_artist.get_label()
                    if (
                        isinstance(label_value, str)
                        and label_value
                        and not label_value.startswith("_")
                        and rolling_artist.get_visible()
                    ):
                        legend_handles_found = True

        # Only install date machinery when the axis really is temporal: on a
        # numeric axis AutoDateLocator reads the values as days since the epoch,
        # so a plain 0..4000 x range was labelled "1970 ... 1980".
        if any_temporal_axis:
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

        if legend_handles_found:
            ax.legend()
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})


    def _draw_colored_markers(
        self,
        *,
        ax,
        frame: pd.DataFrame,
        marker_kwargs: dict[str, Any],
        fallback_color: Any,
        color_is_discrete: bool,
    ) -> None:
        """Draw markers colored from the optional ``color`` role."""
        valid = frame["x"].notna() & frame["y"].notna()
        points = frame.loc[valid]
        if points.empty:
            return

        kwargs = dict(marker_kwargs)
        color_values = points["color"]
        if color_is_discrete:
            colors = self.map_integer_colors_to_palette(color_values, fallback_color)
            kwargs.pop("c", None)
            kwargs.pop("color", None)
            kwargs.pop("facecolor", None)
            kwargs.pop("facecolors", None)
            kwargs.pop("edgecolor", None)
            kwargs.pop("edgecolors", None)
            kwargs.pop("cmap", None)
            kwargs.pop("norm", None)
            kwargs.pop("vmin", None)
            kwargs.pop("vmax", None)
            kwargs["facecolors"] = colors
            kwargs["edgecolors"] = colors
        else:
            kwargs["c"] = pd.to_numeric(color_values, errors="coerce").to_numpy(dtype=float)

        try:
            ax.scatter(
                points["x"].to_numpy(),
                points["y"].to_numpy(dtype=float),
                **kwargs,
            )
        except Exception:
            applogger.exception("Failed to draw colored time-series markers.")

    def _draw_color_segments(
        self,
        *,
        ax,
        frame: pd.DataFrame,
        line_kwargs: dict[str, Any],
        fallback_color: Any,
        show_legend: bool,
        label: str,
    ) -> bool:
        """Draw contiguous line segments using colors from the ``color`` role.

        Matplotlib ``plot`` supports one color per Line2D. A clustered time
        series needs different colors along the same series, so split the line
        whenever the resolved color changes or a NaN gap is present.
        """
        if frame.empty or "color" not in frame.columns:
            return False

        resolved_colors = self.color_sequence_from_values(
            frame["color"],
            fallback_color=fallback_color,
        )
        remaining_label = label if show_legend and label else "_nolegend_"
        legend_handle_found = False

        start = 0
        n_rows = len(frame)
        while start < n_rows - 1:
            while start < n_rows - 1 and (
                pd.isna(frame.iloc[start]["x"])
                or pd.isna(frame.iloc[start]["y"])
                or pd.isna(frame.iloc[start + 1]["x"])
                or pd.isna(frame.iloc[start + 1]["y"])
            ):
                start += 1
            if start >= n_rows - 1:
                break

            color = resolved_colors[start]
            end = start + 1
            while end < n_rows - 1:
                if (
                    pd.isna(frame.iloc[end]["x"])
                    or pd.isna(frame.iloc[end]["y"])
                    or pd.isna(frame.iloc[end + 1]["x"])
                    or pd.isna(frame.iloc[end + 1]["y"])
                    or resolved_colors[end] != color
                ):
                    break
                end += 1

            segment = frame.iloc[start : end + 1]
            kwargs = dict(line_kwargs)
            kwargs["color"] = color
            kwargs["label"] = remaining_label
            try:
                lines = ax.plot(
                    segment["x"].to_numpy(),
                    segment["y"].to_numpy(dtype=float),
                    **kwargs,
                )
                if lines and remaining_label and not remaining_label.startswith("_"):
                    artist = lines[0]
                    if artist.get_visible():
                        legend_handle_found = True
                remaining_label = "_nolegend_"
            except Exception:
                applogger.exception("Failed to draw colored time-series segment.")

            start = max(end, start + 1)

        return legend_handle_found

    def _series_line_color(self, df: pd.DataFrame, style: dict[str, Any], layer_index: int) -> Any:
        fallback_color = self.series_color(style, layer_index)
        if "color" in df.columns:
            return self.first_color_from_values(df["color"], fallback_color=fallback_color)
        return fallback_color

    def _draw_error_bars(
        self,
        *,
        ax,
        sd: SeriesData,
        mask: pd.Series,
        x_values,
        y_values,
        style: dict[str, Any],
        base_kwargs: dict[str, Any],
        series_index: int,
    ) -> None:
        """Draw x/y error bars for one series, with no marker or line."""
        x_error = self.error_values(sd.df, "x", mask)
        y_error = self.error_values(sd.df, "y", mask)
        if x_error is None and y_error is None:
            return

        error_kwargs = self.error_kwargs(base_kwargs)
        error_kwargs.setdefault(
            "ecolor",
            str(style.get("color", "") or "").strip()
            or self.series_color(style, series_index),
        )
        if "alpha" in style and style["alpha"] not in (None, ""):
            error_kwargs["alpha"] = style["alpha"]
        if "zorder" in style and style["zorder"] not in (None, ""):
            error_kwargs["zorder"] = style["zorder"]

        try:
            ax.errorbar(
                x_values,
                y_values,
                xerr=x_error,
                yerr=y_error,
                fmt="none",
                label="_nolegend_",
                **error_kwargs,
            )
        except Exception:
            applogger.exception(
                "Failed to draw error bars for series '%s'.", sd.name
            )

    # The rule itself lives in utils.coercion, because the outlier and
    # clustering dialogs need the same one and had a broken copy.
    _coerce_x_axis = staticmethod(coerce_axis)

    @staticmethod
    def _insert_gaps(
        frame: pd.DataFrame,
        threshold: str,
        *,
        temporal: bool = True,
    ) -> pd.DataFrame:
        """Insert NaN rows before points separated by a large gap in x.

        This breaks the plotted line visually where discontinuities exceed the
        given threshold.  On a numeric axis the threshold is read as a plain
        number; on a time axis it is a pandas offset string such as "2h".
        """
        if frame.empty:
            return frame

        result = frame.copy()

        if temporal:
            result["x"] = pd.to_datetime(result["x"], errors="coerce")
            try:
                thr = pd.Timedelta(threshold)
            except ValueError:
                applogger.warning(
                    "Invalid time gap threshold %r; gaps disabled.",
                    threshold,
                    show_dialog=False,
                    raise_error=False,
                )
                return result
        else:
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                # A time offset like "2h" on a numeric axis means "no gaps",
                # which is the safe reading: never split a line by accident.
                return result

        dt = result["x"].diff()
        gap_positions = np.flatnonzero((dt > thr).fillna(False).to_numpy())

        if gap_positions.size == 0:
            return result

        parts: list[pd.DataFrame] = []
        start = 0

        for idx in gap_positions:
            parts.append(result.iloc[start:idx])

            parts.append(
                pd.DataFrame(
                    {
                        "x": [result.iloc[idx]["x"]],
                        "y": [np.nan],
                    }
                )
            )
            start = idx

        parts.append(result.iloc[start:])
        return pd.concat(parts, ignore_index=True)
