"""Render a figure descriptor tree onto a Matplotlib figure.

Responsibilities and non-responsibilities are deliberately split: this module
creates axes, loads series data, dispatches to renderers, and applies layout.
It does **not** own figure width, height, or DPI - those belong to ChartPanel
and to rcParams, so the same descriptor renders identically on screen and on
export.

Series data is read through ``SqliteRepo.series_df`` so that repeated renders of
an unchanged database are served from cache.
"""
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from app.scanners.axis_renderer_scanner import get_renderer, import_class_from_file
from app.charts.base import SeriesData
from app.data.descriptors import AxisDescriptor, FigureDescriptor, SeriesDescriptor
from app.data.sqlite_repo import SqliteRepo
from app.logs.logger import applogger
from app.utils.mpl_latex import filter_latex_style_text


LayoutMode = Literal["constrained", "compressed", "tight", "none"]


# ----------------------------------------------------------------------
# Context helpers
# ----------------------------------------------------------------------
@contextmanager
def _figure_style_context(
    fig_desc: FigureDescriptor,
) -> Generator[None, None, None]:
    """Apply raw mplstyle text from figure options, if present.

    LaTeX-dependent entries are stripped when no TeX installation is available,
    otherwise every text draw in the figure would raise.
    """
    options = fig_desc.options if isinstance(fig_desc.options, dict) else {}
    style_text = options.get("mpl_style")

    if not isinstance(style_text, str) or not style_text.strip():
        yield
        return

    style_text = filter_latex_style_text(style_text)

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            suffix=".mplstyle",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write(style_text)
            temp_path = Path(handle.name)

        with plt.style.context(str(temp_path)):
            yield
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Public renderer entry point
# ----------------------------------------------------------------------
def render_figure_from_descriptor(
    *,
    figure: Figure,
    descriptor: FigureDescriptor,
    repo: SqliteRepo,
) -> None:
    """Render a full figure from a project descriptor.

    The renderer deliberately does not own figure width, height, or DPI.  Those
    values are owned by ChartPanel and rcParams.  This function only creates
    axes, renders series, applies runtime options, and applies final layout.
    """
    figure.clear()
    _set_layout_engine_safely(figure, "none")

    with _figure_style_context(descriptor):
        _apply_figure_options(figure, descriptor)

        axes_with_positions, rows, cols, spans_valid = _normalized_axes_for_grid(descriptor)
        axes_flat = _create_axes_grid(
            figure=figure,
            descriptor=descriptor,
            axes_with_positions=axes_with_positions,
            rows=rows,
            cols=cols,
            respect_spans=spans_valid,
        )

        for axis_desc, axis_index in axes_with_positions:
            ax = axes_flat[axis_index]
            if ax is None:
                applogger.error(
                    "No subplot was created for axis id=%r at render position=%r.",
                    axis_desc.id,
                    axis_index,
                )
                return

            series_list = _build_series_data_list(repo=repo, axis_desc=axis_desc)
            chart_type = str(axis_desc.chart_type or "").strip()
            renderer = get_renderer(chart_type)
            if renderer is None:
                applogger.error("Renderer not found for chart_type=%r.", chart_type)
                return

            try:
                renderer_instance = import_class_from_file(renderer)()  # type: ignore
                renderer_instance.render_axis(
                    ax=ax,
                    series=series_list,
                    options=axis_desc.options if isinstance(axis_desc.options, dict) else {},
                )
            except Exception:
                applogger.exception(
                    "Renderer failed for axis id=%r chart_type=%r.",
                    axis_desc.id,
                    chart_type,
                )
                return

            _apply_axis_runtime_options(ax=ax, axis_desc=axis_desc)

        _apply_layout(figure, descriptor)
        _normalize_axes_fill_policy(figure, descriptor)


# ----------------------------------------------------------------------
# Axes creation / normalization
# ----------------------------------------------------------------------
def _compact_grid_for_axis_count(axis_count: int) -> tuple[int, int]:
    """Return a compact grid that can contain axis_count axes."""
    axis_count = max(1, int(axis_count))
    if axis_count == 1:
        return 1, 1

    ncols = int(math.ceil(math.sqrt(axis_count)))
    nrows = int(math.ceil(axis_count / ncols))
    return max(1, nrows), max(1, ncols)


def _axis_span(axis_desc: AxisDescriptor) -> tuple[int, int]:
    """Return (row_span, col_span) for one axis, at least 1x1.

    Spans let an axis occupy more than one grid cell - the mechanism behind
    every non-uniform layout the Figure options panel offers (one wide plot
    over two narrow ones, an L-shape, and so on). They live in
    ``axis_desc.options`` rather than as descriptor columns because they are
    just another rendering option, no different from a scale or a spine.
    """
    options = axis_desc.options if isinstance(axis_desc.options, dict) else {}
    row_span = options.get("row_span", 1)
    col_span = options.get("col_span", 1)
    try:
        row_span = max(1, int(row_span))
    except (TypeError, ValueError):
        row_span = 1
    try:
        col_span = max(1, int(col_span))
    except (TypeError, ValueError):
        col_span = 1
    return row_span, col_span


def _axis_footprint(
    *, axis_index: int, row_span: int, col_span: int, rows: int, cols: int
) -> set[tuple[int, int]] | None:
    """Return the set of (row, col) cells one axis occupies, or None if it
    does not fit inside the grid starting from its top-left cell."""
    if cols <= 0:
        return None
    row = axis_index // cols
    col = axis_index % cols
    if row + row_span > rows or col + col_span > cols:
        return None
    return {(r, c) for r in range(row, row + row_span) for c in range(col, col + col_span)}


def _normalized_axes_for_grid(
    descriptor: FigureDescriptor,
) -> tuple[list[tuple[AxisDescriptor, int]], int, int, bool]:
    """Return axes with safe zero-based positions, rows, columns, and whether
    their row_span/col_span options can be trusted.

    An axis normally fills exactly one grid cell. ``row_span``/``col_span`` in
    its options let it fill a rectangle of cells instead, so the figure is no
    longer limited to a uniform grid - e.g. one axis spanning the whole top
    row with two narrower axes below it. Positions and spans are validated
    together: if any axis would fall outside the grid or overlap another
    axis's footprint, every axis falls back to one cell each in a compact
    grid, exactly as an out-of-range axis_index already did. The returned
    ``spans_valid`` flag tells the caller whether that fallback happened -
    when it did, the stale row_span/col_span values in the (now repositioned)
    axes must be ignored rather than reapplied to their new, smaller cells.
    """
    axes = list(descriptor.axes or [])

    base_rows = max(1, int(descriptor.nrows or 1))
    base_cols = max(1, int(descriptor.ncols or 1))
    base_total = base_rows * base_cols

    if not axes:
        return [], base_rows, base_cols, True

    rows = base_rows
    cols = base_cols
    if base_total < len(axes):
        rows, cols = _compact_grid_for_axis_count(len(axes))

    used: set[tuple[int, int]] = set()
    valid_layout = True

    for axis_desc in axes:
        axis_index = int(axis_desc.axis_index)
        if axis_index < 0 or axis_index >= rows * cols:
            valid_layout = False
            break
        row_span, col_span = _axis_span(axis_desc)
        footprint = _axis_footprint(
            axis_index=axis_index, row_span=row_span, col_span=col_span, rows=rows, cols=cols
        )
        if footprint is None or footprint & used:
            valid_layout = False
            break
        used |= footprint

    if valid_layout:
        return [(axis_desc, int(axis_desc.axis_index)) for axis_desc in axes], rows, cols, True

    ordered_axes = sorted(
        axes,
        key=lambda axis_desc: (int(axis_desc.axis_index), int(axis_desc.id)),
    )
    rows, cols = _compact_grid_for_axis_count(len(ordered_axes))
    normalized = [(axis_desc, index) for index, axis_desc in enumerate(ordered_axes)]

    applogger.warning(
        "Normalized stale axis indexes/spans for figure id=%r: grid %sx%s, axes=%s",
        descriptor.id,
        rows,
        cols,
        [(int(axis.id), int(axis.axis_index), new_index) for axis, new_index in normalized],
    )

    return normalized, rows, cols, False


def _subplot_kwargs_for_axis(
    *,
    axis_desc: AxisDescriptor,
    first_ax: Any | None,
    created_by_id: dict[int, Any],
) -> dict[str, Any]:
    """Build add_subplot keyword arguments for one axis."""
    options = axis_desc.options if isinstance(axis_desc.options, dict) else {}
    subplot = options.get("subplot", {})
    subplot = subplot if isinstance(subplot, dict) else {}

    kwargs: dict[str, Any] = {}

    projection = subplot.get("projection") or options.get("projection")
    if isinstance(projection, str) and projection.strip():
        kwargs["projection"] = projection.strip()

    sharex = subplot.get("sharex", options.get("sharex"))
    sharey = subplot.get("sharey", options.get("sharey"))

    if isinstance(sharex, int) and sharex in created_by_id:
        kwargs["sharex"] = created_by_id[sharex]
    elif sharex is True and first_ax is not None:
        kwargs["sharex"] = first_ax

    if isinstance(sharey, int) and sharey in created_by_id:
        kwargs["sharey"] = created_by_id[sharey]
    elif sharey is True and first_ax is not None:
        kwargs["sharey"] = first_ax

    if bool(options.get("polar", False)):
        kwargs["projection"] = "polar"

    return kwargs


def _create_axes_grid(
    *,
    figure: Figure,
    descriptor: FigureDescriptor,
    axes_with_positions: list[tuple[AxisDescriptor, int]],
    rows: int,
    cols: int,
    respect_spans: bool = True,
) -> list[Any]:
    """Create all axes from a GridSpec using normalized positions.

    A GridSpec is used instead of the plain ``add_subplot(rows, cols, n)``
    numbering so that an axis whose options carry ``row_span``/``col_span``
    can occupy a rectangle of cells rather than exactly one - the mechanism
    that lets a figure use a layout other than a uniform grid.  A 1x1 span
    behaves exactly like the old numbered subplot, so every figure without
    spans renders identically to before.

    ``respect_spans=False`` forces every axis to one cell.  ``_normalized_
    axes_for_grid`` sets this when it had to fall back to a compact grid: the
    axes were repositioned into smaller cells that their original spans no
    longer fit, so honouring stale span values here would overlap axes again
    instead of resolving the conflict that triggered the fallback.
    """
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    total = rows * cols

    axes_flat: list[Any] = [None] * total
    created_by_id: dict[int, Any] = {}
    first_ax: Any | None = None
    gridspec = figure.add_gridspec(rows, cols)

    for axis_desc, axis_index in axes_with_positions:
        axis_index = int(axis_index)
        if axis_index < 0 or axis_index >= total:
            applogger.error(
                "Axis id=%r normalized to invalid position %r for grid %sx%s.",
                axis_desc.id,
                axis_index,
                rows,
                cols,
            )
            return axes_flat

        axis_id = int(axis_desc.id)
        row = axis_index // cols
        col = axis_index % cols
        row_span, col_span = _axis_span(axis_desc) if respect_spans else (1, 1)
        # Defensive clamp: normalization already validates footprints for the
        # common case, but a span edited directly in a project file should
        # degrade to a smaller subplot rather than raise.
        row_span = max(1, min(row_span, rows - row))
        col_span = max(1, min(col_span, cols - col))

        kwargs = _subplot_kwargs_for_axis(
            axis_desc=axis_desc,
            first_ax=first_ax,
            created_by_id=created_by_id,
        )

        ax = figure.add_subplot(
            gridspec[row : row + row_span, col : col + col_span], **kwargs
        )
        axes_flat[axis_index] = ax
        created_by_id[axis_id] = ax

        if first_ax is None:
            first_ax = ax

    return axes_flat


# ----------------------------------------------------------------------
# Series loading
# ----------------------------------------------------------------------
def _build_series_data_list(
    *,
    repo: SqliteRepo,
    axis_desc: AxisDescriptor,
) -> list[SeriesData]:
    """Build the list of visible series for one axis."""
    output: list[SeriesData] = []

    for series_desc in list(axis_desc.series or []):
        style = _series_style(series_desc)
        if not bool(style.get("visible", True)):
            continue

        df = _load_series_df(repo=repo, series_desc=series_desc)
        output.append(
            SeriesData(
                name=str(series_desc.name or ""),
                df=df,
                style=style,
            )
        )

    return output


def _load_series_df(
    *,
    repo: SqliteRepo,
    series_desc: SeriesDescriptor,
) -> pd.DataFrame:
    """Load one series DataFrame from its SQL query.

    Goes through ``SqliteRepo.series_df`` so repeated renders of an unchanged
    database are served from cache: SQLite row materialisation is ~89 % of the
    render cost and cannot be optimised away in pure Python.
    """
    sql = str(series_desc.sql_query or "").strip()
    if not sql:
        applogger.error(
            "SQL query is empty for series id=%r name=%r.",
            series_desc.id,
            series_desc.name,
        )
        return pd.DataFrame()

    try:
        return repo.series_df(sql)
    except Exception:
        applogger.exception(
            "Failed to load SQL data for series id=%r name=%r.",
            series_desc.id,
            series_desc.name,
        )
        return pd.DataFrame()


def _series_style(series_desc: SeriesDescriptor) -> dict[str, Any]:
    """Return a normalized series style dictionary."""
    style = series_desc.style
    if isinstance(style, dict):
        return dict(style)
    return {}


# ----------------------------------------------------------------------
# Figure / axis options
# ----------------------------------------------------------------------
def _apply_figure_options(figure: Figure, fig_desc: FigureDescriptor) -> None:
    """Apply figure-level options that do not own physical metrics.

    Figure face and edge colours are re-read from the active rcParams here.
    Why: Matplotlib freezes those two on the Figure at construction time, and
    ChartPanel constructs its Figure once and reuses it for the life of the
    panel, so ``figure.facecolor`` in a user's style file would otherwise have
    no visible effect at all.
    """
    options = fig_desc.options if isinstance(fig_desc.options, dict) else {}

    try:
        figure.set_facecolor(plt.rcParams["figure.facecolor"])
        figure.set_edgecolor(plt.rcParams["figure.edgecolor"])
    except Exception:
        applogger.exception("Failed to apply figure colours from rcParams")

    try:
        figure.set_frameon(bool(options.get("frameon", True)))
    except Exception:
        applogger.exception("Failed to apply figure frameon")

    suptitle = options.get("suptitle")
    if isinstance(suptitle, str) and suptitle.strip():
        figure.suptitle(suptitle.strip())


def _apply_axis_runtime_options(ax: Any, axis_desc: AxisDescriptor) -> None:
    """Apply axis options after the chart renderer has drawn the axis."""
    options = axis_desc.options if isinstance(axis_desc.options, dict) else {}

    title = options.get("title") or options.get("label") or axis_desc.title
    if isinstance(title, str) and title.strip():
        ax.set_title(title.strip())

    xlabel = options.get("x_label", axis_desc.x_label)
    ylabel = options.get("y_label", axis_desc.y_label)
    zlabel = options.get("z_label", axis_desc.z_label)

    # Only non-empty labels are applied.  Why: some renderers set a meaningful
    # default label themselves (the histogram labels its count axis), and an
    # empty descriptor field means "nothing specified", not "clear it".
    if isinstance(xlabel, str) and xlabel.strip():
        ax.set_xlabel(xlabel)
    if isinstance(ylabel, str) and ylabel.strip():
        ax.set_ylabel(ylabel)
    if isinstance(zlabel, str) and zlabel.strip() and hasattr(ax, "set_zlabel"):
        ax.set_zlabel(zlabel)

    if bool(options.get("hide_axis", options.get("hidden", False))):
        ax.set_axis_off()
    else:
        ax.set_axis_on()

    # Order is not free here:
    #   scale -> limits -> inversion.
    # set_xscale() resets the view interval, so it has to run before the limits
    # are applied, and set_xlim() rewrites the direction, so inversion has to
    # run after them.  Applying these in any other order silently loses one of
    # the three settings.
    _apply_scales(ax, options)
    _apply_limits(ax, options)
    _apply_inversion(ax, options)

    _apply_ticks(ax, options)
    _apply_grid(ax, options)
    _apply_spines(ax, options)
    _apply_pickradius(ax, options)
    _apply_aspect_options(ax, options)


# Matplotlib scales that need no extra arguments beyond the base.
SUPPORTED_AXIS_SCALES: tuple[str, ...] = ("linear", "log", "symlog", "logit")
TICK_DIRECTIONS: tuple[str, ...] = ("in", "out", "inout")
GRID_AXES: tuple[str, ...] = ("both", "x", "y")
GRID_WHICH: tuple[str, ...] = ("major", "minor", "both")


def _apply_scales(ax: Any, options: dict[str, Any]) -> None:
    """Apply per-axis scale, with a log base where the scale accepts one."""
    for axis_name, setter_name in (("x", "set_xscale"), ("y", "set_yscale")):
        raw = options.get(f"{axis_name}_scale", "linear")
        scale = str(raw or "linear").strip().lower()
        if scale == "linear":
            continue

        if scale not in SUPPORTED_AXIS_SCALES:
            applogger.warning("Unknown %s scale: %r", axis_name, raw)
            continue

        setter = getattr(ax, setter_name, None)
        if setter is None:
            continue

        kwargs: dict[str, Any] = {}
        if scale in {"log", "symlog"}:
            base = options.get(f"{axis_name}_scale_base", options.get("scale_base"))
            if base is not None:
                try:
                    base_value = float(base)
                    if base_value > 1.0:
                        kwargs["base"] = base_value
                except (TypeError, ValueError):
                    applogger.warning("Invalid log base for %s axis: %r", axis_name, base)
        if scale == "symlog":
            threshold = options.get(f"{axis_name}_linthresh", options.get("linthresh"))
            if threshold is not None:
                try:
                    threshold_value = float(threshold)
                    if threshold_value > 0.0:
                        kwargs["linthresh"] = threshold_value
                except (TypeError, ValueError):
                    applogger.warning("Invalid linthresh for %s axis: %r", axis_name, threshold)

        try:
            setter(scale, **kwargs)
        except Exception:
            applogger.exception("Failed to set %s scale to %r", axis_name, scale)


def _apply_inversion(ax: Any, options: dict[str, Any]) -> None:
    """Point each axis in the requested direction.

    Uses the absolute ``set_inverted`` rather than ``invert_xaxis``: the latter
    toggles, so re-rendering an already inverted axis would flip it back.
    """
    for axis_name, axis_object in (("x", ax.xaxis), ("y", ax.yaxis)):
        wanted = bool(options.get(f"invert_{axis_name}", False))
        try:
            if bool(axis_object.get_inverted()) != wanted:
                axis_object.set_inverted(wanted)
        except Exception:
            applogger.exception("Failed to set %s axis direction", axis_name)


def _apply_ticks(ax: Any, options: dict[str, Any]) -> None:
    """Apply minor ticks, tick direction, size, and label rotation."""
    if bool(options.get("minor_ticks", False)):
        try:
            ax.minorticks_on()
        except Exception:
            applogger.exception("Failed to enable minor ticks")

    params: dict[str, Any] = {}

    direction = str(options.get("tick_direction", "") or "").strip().lower()
    if direction:
        if direction in TICK_DIRECTIONS:
            params["direction"] = direction
        else:
            applogger.warning("Unknown tick direction: %r", direction)

    length = options.get("tick_length")
    if length is not None:
        try:
            params["length"] = float(length)
        except (TypeError, ValueError):
            applogger.warning("Invalid tick length: %r", length)

    if params:
        try:
            ax.tick_params(axis="both", which="both", **params)
        except Exception:
            applogger.exception("Failed to apply tick parameters")

    rotation = options.get("x_tick_rotation")
    if rotation is not None:
        try:
            rotation_value = float(rotation)
        except (TypeError, ValueError):
            applogger.warning("Invalid x tick rotation: %r", rotation)
        else:
            for label in ax.get_xticklabels():
                label.set_rotation(rotation_value)
                # Rotated labels overlap the axis unless they are anchored at
                # the end that stays next to the tick.
                if rotation_value:
                    label.set_horizontalalignment("right")


def _apply_grid(ax: Any, options: dict[str, Any]) -> None:
    """Apply the grid, including which ticks and which axes it follows."""
    if not bool(options.get("grid", False)):
        return

    which = str(options.get("grid_which", "major") or "major").strip().lower()
    if which not in GRID_WHICH:
        applogger.warning("Unknown grid_which: %r", which)
        which = "major"

    axis = str(options.get("grid_axis", "both") or "both").strip().lower()
    if axis not in GRID_AXES:
        applogger.warning("Unknown grid_axis: %r", axis)
        axis = "both"

    # A minor grid without minor ticks draws nothing, which reads as a broken
    # setting rather than an empty one.
    if which in {"minor", "both"}:
        try:
            ax.minorticks_on()
        except Exception:
            applogger.exception("Failed to enable minor ticks for the minor grid")

    try:
        ax.grid(True, which=which, axis=axis)
    except Exception:
        applogger.exception("Failed to apply the axis grid")


def _apply_spines(ax: Any, options: dict[str, Any]) -> None:
    """Show or hide individual spines."""
    spines = getattr(ax, "spines", None)
    if spines is None:
        return

    for name in ("top", "right", "bottom", "left"):
        key = f"hide_spine_{name}"
        if key not in options:
            continue
        try:
            spines[name].set_visible(not bool(options.get(key)))
        except Exception:
            applogger.exception("Failed to set visibility of the %s spine", name)


def _apply_limits(ax: Any, options: dict[str, Any]) -> None:
    """Apply optional axis limits."""
    xlim = options.get("xlim")
    ylim = options.get("ylim")

    if isinstance(xlim, (list, tuple)) and len(xlim) == 2:
        try:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        except (TypeError, ValueError):
            applogger.warning("Invalid xlim: %r", xlim)

    if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
        try:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        except (TypeError, ValueError):
            applogger.warning("Invalid ylim: %r", ylim)


def _apply_pickradius(ax: Any, options: dict[str, Any]) -> None:
    """Apply optional pickradius if supported by the axis."""
    pickradius = options.get("pickradius")
    if pickradius is None:
        return

    try:
        ax.pickradius = float(pickradius)
    except (AttributeError, TypeError, ValueError):
        applogger.info("Ignoring unsupported pickradius=%r", pickradius)


def _apply_aspect_options(ax: Any, options: dict[str, Any]) -> None:
    """Apply optional aspect and adjustable axis options."""
    aspect = options.get("aspect")
    if aspect is not None:
        try:
            if isinstance(aspect, str):
                ax.set_aspect(aspect.strip())
            else:
                ax.set_aspect(float(aspect))
        except (TypeError, ValueError):
            applogger.warning("Invalid axis aspect: %r", aspect)

    adjustable = options.get("adjustable")
    if isinstance(adjustable, str) and adjustable.strip():
        try:
            ax.set_adjustable(adjustable.strip())
        except ValueError:
            applogger.warning("Invalid axis adjustable value: %r", adjustable)


def _apply_layout(figure: Figure, fig_desc: FigureDescriptor) -> None:
    """Apply final figure layout.

    Figure option example:
        {
            "layout_mode": "none",
            "margins": {
                "left": 0.04,
                "right": 0.99,
                "bottom": 0.06,
                "top": 0.97,
                "wspace": 0.08,
                "hspace": 0.08
            }
        }
    """
    options = fig_desc.options if isinstance(fig_desc.options, dict) else {}
    raw_mode = str(options.get("layout_mode", "none") or "none").strip().lower()

    if raw_mode == "constrained":
        _set_layout_engine_safely(figure, "constrained")
        return

    if raw_mode == "compressed":
        _set_layout_engine_safely(figure, "compressed")
        return

    if raw_mode == "tight":
        _set_layout_engine_safely(figure, "tight")
        return

    _set_layout_engine_safely(figure, "none")
    _apply_subplot_margins(figure, options)


def _apply_subplot_margins(figure: Figure, options: dict[str, Any]) -> None:
    """Apply explicit subplot margins for layout_mode='none'."""
    margins = options.get("margins")
    margins = margins if isinstance(margins, dict) else {}

    left = _float_option(margins, "left", 0.04)
    right = _float_option(margins, "right", 0.99)
    bottom = _float_option(margins, "bottom", 0.06)
    top = _float_option(margins, "top", 0.97)
    wspace = _float_option(margins, "wspace", 0.08)
    hspace = _float_option(margins, "hspace", 0.08)

    try:
        figure.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
            hspace=hspace,
        )
    except Exception:
        applogger.exception("Failed to apply subplot margins")


def _normalize_axes_fill_policy(figure: Figure, fig_desc: FigureDescriptor) -> None:
    """Prevent fixed-aspect axes from shrinking their axes box by default."""
    fig_options = fig_desc.options if isinstance(fig_desc.options, dict) else {}
    preserve_all = bool(fig_options.get("preserve_aspect_box", False))

    axes_options = [
        axis_desc.options if isinstance(axis_desc.options, dict) else {}
        for axis_desc in list(fig_desc.axes or [])
    ]

    for index, ax in enumerate(list(figure.axes)):
        try:
            axis_options = axes_options[index] if index < len(axes_options) else {}
            if preserve_all or bool(axis_options.get("preserve_aspect_box", False)):
                continue

            if ax.get_aspect() != "auto" and ax.get_adjustable() == "box":
                ax.set_adjustable("datalim")

            ax.set_anchor("C")
        except Exception:
            applogger.exception("Failed to normalize axes fill policy")


def _set_layout_engine_safely(figure: Figure, mode: LayoutMode) -> None:
    """Set Matplotlib layout engine without letting layout failure stop render."""
    try:
        figure.set_layout_engine(mode)
    except Exception:
        applogger.exception("Failed to set Matplotlib layout engine: %s", mode)


def _float_option(options: dict[str, Any], key: str, default: float) -> float:
    """Read a float option safely."""
    value = options.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        applogger.warning("Invalid float option %s=%r, using %r", key, value, default)
        return default
