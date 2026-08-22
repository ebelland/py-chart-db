"""Scatter plot renderer.

Draws one collection per series, optionally colouring and sizing points from
extra role columns.  Integer colour columns are treated as discrete category
ids; float columns go through a colormap.
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D

from app.charts.base_axis import ERROR_BAR_KWARGS, BaseAxisRenderer, SeriesData
from app.logs.logger import applogger


def _reorder_error(error: Any, order: np.ndarray) -> Any:
    """Apply a point ordering to a symmetric or asymmetric error array.

    Symmetric errors are one value per point; asymmetric ones are (2, N), so
    the reordering has to happen along the last axis in both cases.
    """
    if error is None:
        return None
    array = np.asarray(error)
    return array[..., order]



class ScatterAxisRenderer(BaseAxisRenderer):
    """Renderer for 2D scatter plots.

    Behavior:
    - If marker is set, draw scatter markers.
    - If line style is set, draw a connecting line.
    - If marker is empty and line style is set, draw only the line.
    - If both marker and line style are empty/none, nothing is drawn.

    Honors per-series style fields such as:
        - label
        - visible
        - show_in_legend
        - sort_x
        - color
        - marker
        - linestyle
        - alpha
        - zorder
    """

    RequiredRoles = ["x", "y"]
    OptionalRoles = [
        "color",
        "size",
        # Error bars: one column for a symmetric half-width, or a low/high pair
        # for an asymmetric interval.  See BaseAxisRenderer.error_values.
        "xerr",
        "xerr_low",
        "xerr_high",
        "yerr",
        "yerr_low",
        "yerr_high",
    ]

    Name = "Scatter Plot"
    Category: str = "Pairwise data"
    Description = "Scatter Plot"
    Link:str="https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter"

    def _float_option(self, name: str, options: dict[str, Any], default: float = 0.0) -> float:
        """Return a renderer option as float without upsetting Pylance.

        BaseAxisRenderer.opt() is intentionally generic and returns object, so
        direct calls like float(self.opt(...)) are reported by Pylance.  This
        wrapper narrows the value at runtime and gives the type checker a real
        float return type.
        """
        value = cast(Any, self.opt(name, options))
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            applogger.warning("Invalid float option %s=%r; using %r", name, value, default)
            return default

    def _int_option(self, name: str, options: dict[str, Any], default: int = 0) -> int:
        """Return a renderer option as int without Pylance object warnings."""
        value = cast(Any, self.opt(name, options))
        if value in (None, ""):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            applogger.warning("Invalid integer option %s=%r; using %r", name, value, default)
            return default

    #: Options this renderer draws itself rather than forwarding to
    #: ax.scatter, which would reject them as unknown keywords.
    OVERLAY_KWARGS: frozenset[str] = frozenset(
        {
            "confidence_ellipse",
            "confidence_ellipse_color",
            "trend_degree",
            "trend_band",
            "trend_sigma",
        }
    )

    Kwargs: dict[str, object] = {
        "confidence_ellipse": {
            "default": 0.0,
            "type": float,
            "min": 0.0,
            "max": 5.0,
            "group": "Statistics",
            "description": (
                "Draw the covariance ellipse of the cloud, this many standard "
                "deviations across. 0 draws none; 2 covers about 95% of a "
                "bivariate normal sample. It shows the correlation as a tilt, "
                "which a cloud of points alone does not make obvious."
            ),
        },
        "confidence_ellipse_color": {
            "default": "#D32F2F",
            "type": str,
            "kind": "color",
            "group": "Statistics",
            "description": "Outline colour of the covariance ellipse.",
        },
        "trend_degree": {
            "default": 0,
            "type": int,
            "min": 0,
            "max": 6,
            "group": "Statistics",
            "description": (
                "Degree of a least-squares trend line through the points. "
                "0 draws none, 1 is a straight line."
            ),
        },
        "trend_band": {
            "default": "none",
            "type": ["none", "confidence", "prediction"],
            "group": "Statistics",
            "description": (
                "Uncertainty band around the trend. 'confidence' is the "
                "uncertainty of the fitted curve itself and is narrow; "
                "'prediction' is where the next observation would fall and is "
                "much wider. They answer different questions and are routinely "
                "confused, so they are named rather than offered as one band."
            ),
        },
        "trend_sigma": {
            "default": 2.0,
            "type": float,
            "min": 0.5,
            "max": 5.0,
            "group": "Statistics",
            "description": "Half-width of the band, in standard deviations.",
        },
        "alpha": {
            "default": 0.8,
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "description": (
                "Marker opacity from 0.0 (transparent) to 1.0 (opaque)."
            ),
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
                "Lower bound for colormap normalization "
                "when numeric color data is used."
            ),
        },
        "vmax": {
            "default": None,
            "type": float,
            "min": -1_000_000_000.0,
            "max": 1_000_000_000.0,
            "description": (
                "Upper bound for colormap normalization "
                "when numeric color data is used."
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
            "description": "Whether the scatter collection is visible.",
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
            "description": "Rasterize the scatter collection during vector export.",
        },
        **ERROR_BAR_KWARGS,
    }

    def render_axis(self, ax, series: list[SeriesData], options: dict) -> None:
        """Render all scatter series onto a single axis."""
        base_kwargs = self.get_kwargs(options)

        # ax.scatter and ax.plot reject errorbar keywords, so the drawing
        # kwargs are the base set minus those; the error bars get them back
        # through error_kwargs().
        draw_kwargs = {
            key: value
            for key, value in base_kwargs.items()
            if key not in ERROR_BAR_KWARGS and key not in self.OVERLAY_KWARGS
        }
        legend_handles_found = False

        for series_index, sd in enumerate(series):
            style = dict(sd.style or {})
            if not bool(style.get("visible", True)) or not self.ensure_required_roles(sd.df):
                continue

            x = pd.to_numeric( sd.df['x'], errors="coerce")
            y = pd.to_numeric( sd.df['y'], errors="coerce")
            color_source = sd.df['color'] if 'color' in sd.df.columns else None
            color = pd.to_numeric(color_source, errors="coerce") if color_source is not None else None
            size = pd.to_numeric(sd.df['size'], errors="coerce") if 'size' in sd.df.columns else None

            if x is None or y is None:
                applogger.warning(f"Series { sd.name} skipped: required numeric 'x' and 'y' data not found.")
                continue

            mask = x.notna() & y.notna()
            if not bool(mask.any()):
                applogger.info("Series '%s' skipped: no finite x/y pairs.", sd.name)
                continue

            x_values = x[mask].to_numpy(dtype=float)
            y_values = y[mask].to_numpy(dtype=float)

            color_values = None
            color_is_discrete = False
            if color is not None and color_source is not None:
                masked_color_source = color_source[mask]
                color_is_discrete = self.is_discrete_integer_color(masked_color_source)
                color_values = color[mask].to_numpy(dtype=float)

            size_values = None
            if size is not None:
                size_values = size[mask].to_numpy(dtype=float)

            # Optional sort by X ascending.
            order = np.arange(x_values.size)
            if bool(style.get("sort_x", False)):
                order = np.argsort(x_values, kind="stable")
                x_values = x_values[order]
                y_values = y_values[order]
                if color_values is not None:
                    color_values = color_values[order]
                if size_values is not None:
                    size_values = size_values[order]

            show_in_legend = bool(style.get("show_in_legend", True))
            
            series_label = str(style.get("label", "") or "").strip()
            if not series_label:
                axis_label = str(self.opt("label", options) or "").strip()
                if axis_label.strip():
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

            has_marker = marker_style and marker_style!= ""
            has_line = line_style and line_style.lower() != ""

            # Nothing to draw for this series.
            if not has_marker and not has_line:
                applogger.warning(
                    "Series '%s': ",
                    sd.name,
                )
                marker_style="."
                has_marker=True

            flat_color = str(style.get("color", "") or "").strip()

            # ----------------------------------------------------------
            # Error bars
            # ----------------------------------------------------------
            # Drawn first, and with no marker or line of their own, so the
            # markers below sit on top of them and only one artist per series
            # ends up in the legend.
            if self.has_error_roles(sd.df):
                x_error = self.error_values(sd.df, "x", mask)
                y_error = self.error_values(sd.df, "y", mask)

                if x_error is not None or y_error is not None:
                    if bool(style.get("sort_x", False)):
                        x_error = _reorder_error(x_error, order)
                        y_error = _reorder_error(y_error, order)

                    error_kwargs = self.error_kwargs(base_kwargs)
                    error_kwargs.setdefault(
                        "ecolor", flat_color or self.series_color(style, series_index)
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

            # ----------------------------------------------------------
            # Scatter markers (only if marker is enabled)
            # ----------------------------------------------------------
            if has_marker:
                scatter_kwargs: dict[str, Any] = dict(draw_kwargs)
                scatter_kwargs["marker"] = marker_style

                if "alpha" in style and style["alpha"] not in (None, ""):
                    scatter_kwargs["alpha"] = style["alpha"]

                if "zorder" in style and style["zorder"] not in (None, ""):
                    scatter_kwargs["zorder"] = style["zorder"]

                if "visible" in style:
                    scatter_kwargs["visible"] = bool(style["visible"])

                if color_values is not None:
                    if color_is_discrete:
                        # Discrete integer values are category ids. Resolve them
                        # to explicit face colors before calling scatter so
                        # Matplotlib does not try to parse the integer ids as a
                        # continuous ``c`` array.
                        discrete_colors = self.map_integer_colors_to_palette(color_values, flat_color)
                        # For discrete colors, bypass scatter's ``c`` parsing and
                        # set both face and edge colors explicitly. This avoids
                        # colormap behavior and prevents an axis-level ``c`` or
                        # ``facecolors='none'`` from making only the marker edge
                        # appear colored.
                        scatter_kwargs.pop("c", None)
                        scatter_kwargs.pop("color", None)
                        scatter_kwargs.pop("facecolor", None)
                        scatter_kwargs.pop("facecolors", None)
                        scatter_kwargs.pop("edgecolor", None)
                        scatter_kwargs.pop("edgecolors", None)
                        scatter_kwargs.pop("cmap", None)
                        scatter_kwargs.pop("norm", None)
                        scatter_kwargs.pop("vmin", None)
                        scatter_kwargs.pop("vmax", None)
                        scatter_kwargs["facecolors"] = discrete_colors
                        scatter_kwargs["edgecolors"] = discrete_colors
                    else:
                        scatter_kwargs["c"] = color_values

                        norm = self.opt("norm", options)
                        if norm not in (None, ""):
                            scatter_kwargs["norm"] = norm

                        vmin = self.opt("vmin", options)
                        if vmin not in (None, "") and isinstance(vmin, (int, float)):
                            scatter_kwargs["vmin"] = float(vmin)

                        vmax = self.opt("vmax", options)
                        if vmax not in (None, "") and isinstance(vmax, (int, float)):
                            scatter_kwargs["vmax"] = float(vmax)
                else:
                    if flat_color:
                        scatter_kwargs["c"] = flat_color
                    else:
                        scatter_kwargs.pop("c", None)

                    scatter_kwargs.pop("norm", None)
                    scatter_kwargs.pop("vmin", None)
                    scatter_kwargs.pop("vmax", None)

                if size_values is not None:
                    scatter_kwargs["s"] = size_values
                else:
                    scatter_kwargs.pop("s", None)

                # If a line is also drawn, let the line own the legend label.
                # pop() must tolerate a missing key: get_kwargs only emits keys
                # whose value is not None, so "label" is absent whenever no
                # label was configured.
                if show_in_legend and series_label and series_label != "" and not has_line:
                    scatter_kwargs["label"] = series_label
                else:
                    scatter_kwargs.pop("label", None)
                
                scatter_collection = ax.scatter(
                    x_values,
                    y_values,
                    **scatter_kwargs,
                )

                scatter_label = scatter_collection.get_label()
                if (
                    isinstance(scatter_label, str)
                    and scatter_label
                    and not scatter_label.startswith("_")
                    and scatter_collection.get_visible()
                ):
                    legend_handles_found = True

            # ----------------------------------------------------------
            # Optional connecting line
            # ----------------------------------------------------------
            if has_line:
                line_kwargs: dict[str, Any] = {
                    "linestyle": line_style,
                }

                # If line is drawn together with markers, keep the same marker
                # on the line artist only when marker exists.
  
                line_kwargs["marker"] = "None"

                if flat_color:
                    line_kwargs["color"] = flat_color

                if "alpha" in style and style["alpha"] not in (None, ""):
                    line_kwargs["alpha"] = style["alpha"]

                if "zorder" in style and style["zorder"] not in (None, ""):
                    line_kwargs["zorder"] = style["zorder"]

                linewidth_value = style.get("linewidth", style.get("line_width"))
                if linewidth_value not in (None, ""):
                    line_kwargs["linewidth"] = linewidth_value

                if show_in_legend and series_label:
                    line_kwargs["label"] = series_label
                else:
                    line_kwargs["label"] = "_nolegend_"

                line_list = ax.plot(
                    x_values,
                    y_values,
                    **line_kwargs,
                )
                if line_list:
                    line_artist = line_list[0]
                    line_label = line_artist.get_label()
                    if (
                        isinstance(line_label, str)
                        and line_label
                        and not line_label.startswith("_")
                        and line_artist.get_visible()
                    ):
                        legend_handles_found = True

        # After the points, so the ellipse and the trend sit over the cloud
        # they describe; before the legend, so both can name themselves in it.
        for sd in series:
            if not (sd.style or {}).get("visible", True):
                continue
            if not self.ensure_required_roles(sd.df):
                continue
            x = pd.to_numeric(sd.df["x"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(sd.df["y"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3 and self._draw_overlays(
                ax, x[mask], y[mask], {**options, **(sd.style or {})}
            ):
                legend_handles_found = True

        # points and overlays so the labels/arrows sit above the chart content.

        if legend_handles_found:
            ax.legend()
        # Draw descriptor annotations after renderer-owned artists.
        self.apply_annotations(ax, options or {})

    def _draw_overlays(
        self, ax: Any, x: np.ndarray, y: np.ndarray, options: dict
    ) -> bool:
        """Draw the covariance ellipse and the trend line.  True if either was.

        Both are summaries of the cloud rather than of any one point, which is
        why they are computed here from the finite pairs and not threaded
        through the per-point drawing above.
        """
        drawn = False

        n_std = self._float_option("confidence_ellipse", options, 0.0)
        if n_std > 0.0:
            drawn = self._draw_confidence_ellipse(ax, x, y, n_std, options) or drawn

        degree = self._int_option("trend_degree", options, 0)
        if degree > 0:
            drawn = self._draw_trend(ax, x, y, degree, options) or drawn

        return drawn

    def _draw_confidence_ellipse(
        self, ax: Any, x: np.ndarray, y: np.ndarray, n_std: float, options: dict
    ) -> bool:
        """Draw the covariance ellipse of the cloud.

        Built the way Matplotlib's own example does: a unit circle whose axes
        are set from the Pearson correlation, then rotated 45 degrees and
        scaled by each standard deviation.  The rotation is what encodes the
        correlation - an uncorrelated cloud comes out axis-aligned - and doing
        it through a transform rather than by solving for the eigenvectors
        keeps the ellipse correct when the axes are rescaled under it.
        """
        covariance = np.cov(x, y)
        spread_x, spread_y = covariance[0, 0], covariance[1, 1]
        if not np.isfinite(spread_x) or spread_x <= 0 or spread_y <= 0:
            applogger.info("No covariance ellipse: the cloud has no spread.")
            return False

        pearson = covariance[0, 1] / np.sqrt(spread_x * spread_y)
        pearson = float(np.clip(pearson, -1.0, 1.0))

        ellipse = Ellipse(
            (0.0, 0.0),
            width=np.sqrt(1.0 + pearson) * 2.0,
            height=np.sqrt(1.0 - pearson) * 2.0,
            facecolor="none",
            edgecolor=str(self.opt("confidence_ellipse_color", options) or "#D32F2F"),
            linewidth=1.6,
            linestyle="--",
            label=f"{n_std:g}\u03c3 ellipse",
        )
        ellipse.set_transform(
            Affine2D()
            .rotate_deg(45)
            .scale(np.sqrt(spread_x) * n_std, np.sqrt(spread_y) * n_std)
            .translate(float(np.mean(x)), float(np.mean(y)))
            + ax.transData
        )
        ax.add_patch(ellipse)
        return True

    def _draw_trend(
        self, ax: Any, x: np.ndarray, y: np.ndarray, degree: int, options: dict
    ) -> bool:
        """Draw a least-squares trend line, optionally with its error band."""
        if x.size <= degree + 1:
            applogger.info(
                "No trend line: %d points cannot support a degree-%d fit.",
                x.size,
                degree,
            )
            return False

        coefficients = np.polyfit(x, y, degree)
        grid = np.linspace(float(np.min(x)), float(np.max(x)), 256)
        fitted = np.polyval(coefficients, grid)

        color = str(self.opt("confidence_ellipse_color", options) or "#D32F2F")
        ax.plot(grid, fitted, color=color, linewidth=1.8, label=f"degree {degree} trend")

        band = str(self.opt("trend_band", options) or "none").strip().lower()
        if band in ("confidence", "prediction"):
            residuals = y - np.polyval(coefficients, x)
            # Degrees of freedom, not n: the fit already spent degree+1 of them,
            # and dividing by n would understate the scatter it did not explain.
            dof = max(1, x.size - (degree + 1))
            sigma = float(np.sqrt(np.sum(residuals**2) / dof))

            # The confidence band is the uncertainty of the mean response and
            # narrows towards the centroid; the prediction band adds one whole
            # residual standard deviation, because a future point scatters
            # about the curve as well as the curve being uncertain.
            leverage = 1.0 / x.size + (grid - np.mean(x)) ** 2 / np.sum(
                (x - np.mean(x)) ** 2
            )
            half = sigma * np.sqrt(leverage + (1.0 if band == "prediction" else 0.0))
            half *= self._float_option("trend_sigma", options, 2.0)

            ax.fill_between(
                grid,
                fitted - half,
                fitted + half,
                color=color,
                alpha=0.15,
                linewidth=0,
                label=f"{band} band",
            )
        return True
