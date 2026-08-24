# Developer guide

This document is for anyone changing the code, not using the app. It covers
the architecture, the extension points (new chart type, new series
operation), the figure layout system, styling, localization and testing.

For user-facing behaviour and known issues, see `todo.txt` at the repo root.

## 1. What this app is

A PySide6 desktop front end for building Matplotlib figures from data that
lives in a SQLite database (a `.dhub` file). The three moving parts:

- **Storage** — `app/data/sqlite_repo.py`. Tables/queries the user imports or
  writes, plus three descriptor tables (`__figure_descriptors__`,
  `__axis_descriptors__`, `__series_descriptors__`) that describe how to draw
  them.
- **Render pipeline** — `app/charts/render_figure.py`. Pure function of a
  descriptor tree + a `SqliteRepo` → a populated `matplotlib.figure.Figure`.
  No Qt in this module; it is unit-testable headlessly (see §9).
- **UI** — `app/dialogs/main_window.py` plus the widgets in `app/widgets/`.
  Property panels emit typed payloads (dicts) that the main window persists
  through the repo and then asks the chart panel to redraw.

The loop for "user changes a setting" is always: widget emits a `*_requested`
signal with a plain dict → `MainWindow` merges it into the descriptor's
`options` JSON blob and saves it via `SqliteRepo` → the affected `ChartPanel`
redraws by calling `render_figure_from_descriptor` again. Nothing renders
directly from a widget.

## 2. Project layout

```
app/
  charts/            One module per chart type (renderer), plus
                      render_figure.py (the pipeline) and descriptors.py
                      (the FigureDescriptor/AxisDescriptor/SeriesDescriptor
                      dataclasses).
  data/               SqliteRepo: schema, CRUD, import/export, data sources.
  dialogs/            Top-level QDialog/QMainWindow windows.
  widgets/            Panels embedded in dialogs/the main window
                      (properties editors, table list/preview, chart panel).
  series_operations/  One module per series operation (fit, smoothing,
                      outlier removal, spectral analysis, clustering,
                      calculus, peaks, ...), plus parameter_spec.py /
                      parameter_form.py, the declarative parameter support
                      shared by all of them (see §7.2).
  scanners/           AST-based plugin discovery for renderers and series
                      operations (see §4).
  styles/             style.py (the shared widget/action/icon factory),
                      macos_native.qss, fluent_win11.qss, palettes.py.
  utils/              config.json access, i18n, coercion, messages, dialog
                      state persistence, DPI handling, LaTeX detection,
                      figure_metrics.py (§5.2), series_validation.py (§7.2).
  locales/            gettext-style .po catalogues (see §8).
mplstyles/            The bundled Matplotlib style library (see §7.3).
_make_demo_project.py Builds "Demo Project.dhub" through SqliteRepo - the
                      same code path the app uses, so the demo cannot drift
                      from the real schema. Run it directly to regenerate
                      the demo project after a schema or renderer change.
```

## 3. The descriptor model

Three dataclasses in `app/charts/descriptors.py` mirror the three
descriptor tables one-for-one:

```
FigureDescriptor(id, name, nrows, ncols, options, axes: list[AxisDescriptor])
AxisDescriptor(id, figure_id, axis_index, chart_type, title,
               x_label, y_label, z_label, options, series: list[SeriesDescriptor])
SeriesDescriptor(id, axis_id, series_index, name, sql_query, roles, style)
```

`SqliteRepo.load_figure_descriptor(figure_id)` is the *only* place that
assembles the tree; every other reader goes through it (or through the
narrower `get_axis_options`/`get_*` helpers) rather than querying the tables
directly.

`options` on both `FigureDescriptor` and `AxisDescriptor` is a free-form
`dict[str, Any]`, persisted as one JSON column. This is deliberate: adding a
new figure- or axis-level setting almost never needs a schema migration —
put it in `options`, read it with `.get(key, default)` on the render side,
and add a control that emits it in the matching `*_options_requested`
payload. `MainWindow._on_figure_options_requested` /
`_on_axis_options_requested` persist **the whole payload**, not a hand-listed
subset — see the docstring on the axis handler for why a whitelist there is
a bug waiting to happen.

`roles` on `SeriesDescriptor` maps a renderer's role names (`"x"`, `"y"`,
`"value"`, `"group"`, ...) to columns the series' `sql_query` must produce
via `AS`. Roles are not positional arguments; a renderer declares the
columns it needs (`RequiredRoles`/`OptionalRoles`) and the query is
responsible for aliasing to them.

## 4. Render pipeline (`app/charts/render_figure.py`)

`render_figure_from_descriptor(*, figure, descriptor, repo)` is the entry
point. Order of operations, all of it re-run on every redraw:

1. `figure.clear()`, layout engine reset to `"none"`.
2. `_figure_style_context` — if `options["mpl_style"]` holds raw `.mplstyle`
   text, it is written to a temp file and applied via `plt.style.context(...)`
   for the duration of the render. LaTeX-only entries are stripped first
   (`app/utils/mpl_latex.py`) so a style built with `usetex` doesn't crash on
   a machine without a TeX install.
3. `_apply_figure_options` — face/edge colour (re-read from rcParams, because
   Matplotlib freezes those at Figure construction and ChartPanel's Figure is
   long-lived), `frameon`, `suptitle`.
4. `_normalized_axes_for_grid` + `_create_axes_grid` — turn the axes list into
   a `GridSpec`-backed set of subplots. See §5, this is the layout system.
5. Per axis: load each series' DataFrame through `repo.series_df(sql)`
   (cached — see the module docstring on why this matters for large
   databases), look up the renderer by `chart_type` through the scanner, call
   `renderer.render_axis(ax, series, options)`, then apply the
   renderer-independent runtime options (`_apply_axis_runtime_options`:
   title/labels, scale → limits → inversion **in that order** — see the
   comment above `_apply_axis_runtime_options`, getting this order wrong
   silently drops one of the three — ticks, grid, spines, pick radius,
   aspect).
6. `_apply_layout` — the layout *engine* (`constrained`/`compressed`/`tight`,
   or `"none"` + explicit `subplots_adjust` margins). This is a different
   concept from the grid layout in §5: it controls spacing/margins, not which
   cells axes occupy.
7. `_normalize_axes_fill_policy` — keeps a fixed-aspect axis from shrinking
   its box by default.

The renderer itself never sets labels, scale, grid, or figure size/DPI/frame.
Width, height and DPI are owned by `ChartPanel` + rcParams specifically so a
figure renders identically on screen and on export — see the module
docstring.

## 5. Figure layout: grid and spans

Every figure has an `nrows` × `ncols` grid (`FigureDescriptor.nrows/ncols`,
editable from the Figure panel). By default every axis occupies exactly one
cell, in row-major order given by its `axis_index` (0-based, `row = index //
ncols`, `col = index % ncols`).

An axis can occupy more than one cell by setting `row_span` and/or
`col_span` in its `options` (integers ≥ 1, default 1) — e.g. one axis with
`col_span: 2` in a 2×2 grid spans the whole top row, with two narrower axes
below it. This is exposed in the UI as **Grid position → Span** on the Axis
panel (`app/widgets/axis_properties.py`), one spin box per dimension.
`_make_demo_project.py`'s `_create_layout_showcase_figure` is a worked
example: a scatter plot spanning both columns of a 2×2 grid over a histogram
and a box plot, combined with `frameon: False` and explicit margins.

Internals, in `render_figure.py`:

- `_normalized_axes_for_grid` validates every axis's footprint (its span,
  starting from its `axis_index`) against the grid and against every other
  axis's footprint. If anything is invalid — out of bounds, or two axes'
  footprints overlap — **every** axis in the figure falls back to one cell
  each in a compact grid sized for the axis count, exactly like an
  out-of-range `axis_index` already did before spans existed. This is
  reported back as `spans_valid: bool`.
- `_create_axes_grid` builds one `Figure.add_gridspec(rows, cols)` and slices
  it per axis (`gridspec[row:row+row_span, col:col+col_span]`) instead of the
  old `add_subplot(rows, cols, n)` numbering. When `respect_spans=False` (set
  by the caller whenever normalization had to fall back), every span is
  forced to 1×1 — the axes were just repositioned into smaller cells their
  original span no longer fits, so honouring the stale span would recreate
  the overlap that triggered the fallback in the first place.
- A 1×1 span (the default) produces the exact same geometry as the old
  numbered-subplot code, so no existing figure changes.

Tests: `app/tests/test_figure_layout_and_frame.py`.

### 5.1 Layout engine and manual spacing

Separate from the grid, and easy to confuse with it.
`FigureDescriptor.options["layout_mode"]` picks Matplotlib's **layout
engine**, which decides how the axes the grid produced are spaced:
`constrained`, `compressed`, `tight`, or `none` (shown as **Manual**). Those
four are the whole set `Figure.set_layout_engine()` accepts.

**GridSpec is not a layout engine and does not belong in that list.** It is
how axes are *created* — `_create_axes_from_gridspec` uses it for every
figure — and it is what gives an axis its `row_span`/`col_span` (§5). The two
answer different questions: GridSpec decides where an axis is, the engine
decides how much room is left between them.

Under `none`/Manual, `_apply_subplot_margins` reads
`options["margins"]` — `left`, `right`, `bottom`, `top`, `wspace`, `hspace`
— and applies them via `subplots_adjust`. These are editable as **Manual
spacing** on the Figure panel, enabled only under Manual: the three automatic
engines recompute spacing and overwrite whatever `subplots_adjust` set, so
the controls are disabled with an explanatory tooltip rather than silently
ignored.

`wspace`/`hspace` are fractions of the *average axis size* and legitimately
exceed 1; the four edges are fractions of the figure and do not.

### 5.2 Physical size and DPI

`figure_width_cm`, `figure_height_cm` and `figure_dpi` live in each figure's
`options`, not in `config.json`. They were application-wide until they moved,
which meant opening a second figure applied the first one's size to it. A
slide wants 25cm wide and a journal column wants 8.5.

`app/utils/figure_metrics.py` holds the keys and the reader.
`figure_metrics_from_options()` returns `None` for a figure with no metrics
of its own — deliberately distinct from zero, so a figure saved before the
change keeps rendering as it did instead of being reset to a default. A
half-set of metrics counts as absent too: applying a width without its height
would distort the figure rather than resize it.

`ChartPanel` pushes them into rcParams on construction and on every
`reload()`, because rcParams is process-wide and every figure wants a
different answer.

## 6. `frameon`

`FigureDescriptor.options["frameon"]` (bool, default `True`) is the "Draw
figure frame" checkbox on the Figure panel. It maps directly to
`Figure.set_frameon()` in `_apply_figure_options` — a figure-level setting,
distinct from any axis's own spines (`hide_spine_*` in axis options).

## 7. Extension points

### 7.1 Adding a chart type

Add a module under `app/charts/`, a class implementing
`BaseAxisRenderer` (`app/charts/base.py` — read the protocol docstring
first, it documents the full contract). Nothing else: `axis_renderer_scanner`
AST-scans `app/charts` at import time and finds it by base class name, so a
new file is sufficient — there is no registry to edit.

Minimum surface:

```python
class MyRenderer(BaseAxisRenderer):
    Name = "My Chart"              # shown in the picker; stored as chart_type
    Category = "Pairwise data"     # Matplotlib's own plot-type taxonomy
    Description = "One line for the picker and the axis panel."
    RequiredRoles = ["x", "y"]     # columns the series query must alias to
    OptionalRoles = ["color"]
    Kwargs = {                     # -> auto-generated kwargs editor, no widget code
        "alpha": {"default": 0.8, "type": float, "min": 0.0, "max": 1.0,
                   "group": "Style", "description": "Marker opacity."},
    }

    def render_axis(self, ax, series, options):
        for s in series:
            ax.plot(s.df["x"], s.df["y"], **self.get_kwargs(options))
```

Renaming `Name` later orphans every axis already saved with the old name;
add the rename to `CHART_TYPE_ALIASES` in `axis_renderer_scanner.py` instead
of just changing the string.

Useful shared helpers already on the base class: `series_data_color` /
`series_color` (per-series colour, with an optional `color` data column
overriding the style), `color_sequence_from_values` (integer columns → the
active property cycle, float columns → a colormap), `error_values` /
`error_kwargs` (symmetric or asymmetric `xerr`/`yerr` from `ERROR_BAR_ROLES`
columns), `apply_annotations`.

For x/y/z renderers there is also `app/charts/grids.py`:
`pivot_to_grid(df)` returns `(X, Y, Z)` when the rows form a *complete*
Cartesian product of the distinct x and y values — any spacing, but no
missing pairs — and `None` when they do not, and `finite_xyz(df)` returns
the raw points with the non-finite rows dropped. Call the first one rather
than writing a second grid check: the surface and contour renderers both
split on it, and the two halves of each pair only agree about what "a grid"
means because they ask the same function. The convention when it says no is
to log an error naming the scattered variant and draw nothing, never to
interpolate the missing cells.

A renderer that adds a colorbar must pass `use_gridspec=False` to
`figure.colorbar`. Renderers draw while the figure's layout engine is still
`"none"` — `render_figure` applies the descriptor's layout mode only after
every axis is drawn — and Matplotlib's default colorbar path builds a
`GridSpecFromSubplotSpec` with zero-height padding rows in that state. A
constrained or compressed engine set afterwards then divides by that zero and
the whole figure fails to draw. See `ContourAxisRenderer._colorbar`.

### 7.2 Adding a series operation

A series operation is a self-contained plugin: one file under
`app/series_operations/`, dialog and artwork included (`Icon` is inline SVG
path data on the class, not a file — see `app/series_operations/
dialog_base.py`'s module docstring and any existing
operation for the shape). `app/scanners/series_operation_scanner.py`
discovers it the same way the renderer scanner discovers chart types.
Operations write their result back as a new table prefixed with `_`
(`generated_table_name`) so the source list can group/hide generated tables
separately from imported ones.

#### Declaring parameters instead of building them

Set `PARAMS` on the class and the base builds the form, wires every control
to `refresh_results`, reads the values back by name, and shows or hides each
row — no `build_parameter_selector`, no signal connections, no
`_refresh_visibility`:

```python
PARAMS = (
    FloatParam("threshold", "Threshold:", default_value=3.0,
               minimum=0.1, maximum=30.0,
               visible_for={"model": (OUTLIER_ZSCORE, OUTLIER_MAD)}),
    IntParam("window", "Window size:", default_value=11,
             minimum=3, maximum=9999, odd_only=True,
             visible_for={"model": (OUTLIER_ROLLING,)}),
)
```

Read them with `self.parameter_values()`, which returns **every** declared
name whether or not its row is visible — an operation reading a parameter
belonging to another model gets that parameter's default rather than a
`KeyError`.

- `app/series_operations/parameter_spec.py` — the declarations. Plain data,
  no Qt import, so they stay testable without a window server.
- `app/series_operations/parameter_form.py` — builds the widgets.

`visible_for` maps *another* parameter's name to the values for which this
row is shown. It may name something the form does not own — `model` is the
base's own combo — because `ParameterForm` takes a `context` callable;
`parameter_context()` supplies `model` by default.

`odd_only` matters more than it looks: `savgol_filter` and `medfilt` both
reject an even window with an exception raised from inside SciPy that names
neither the control the user moved nor the series it was moved on.

`PARAMS` is optional. An operation that leaves it empty keeps overriding
`build_parameter_selector` by hand, which is still the right answer for a
genuinely unusual control. The outlier dialog is the converted example.

Honest note on the payoff: converting the outlier dialog changed it from 790
lines to 794. It did not shrink. The gain is in the invariants — visibility
rules and widgets are the same data so they cannot drift, a new parameter
cannot be added without its signal connection, and range clamping happens in
one place — not in the line count.

#### Validating input before the operation runs

Declare what the operation needs of its data and the base checks it:

```python
INPUT_MINIMUM_POINTS = 3
INPUT_REQUIRES_SORTED_X = True
INPUT_REQUIRES_UNIQUE_X = True
INPUT_REQUIRES_UNIFORM_X = False
INPUT_REQUIRES_VARYING_Y = False
```

Then call one of two methods on the materialized arrays:

- `prepare_input_xy(x, y, label=...)` — validates **and repairs**, returning
  cleaned arrays. Drops non-finite points always; sorts and averages
  duplicate x according to the declarations. Every repair is reported.
- `validate_input_xy(x, y, label=..., raise_on_error=False)` — reports only.
  Use this whenever the result is mapped back to source rows: the outlier
  detector matches by rowid and the cluster dialog by frame position, so
  reordering would move each mark onto a different row.

Requirements are declared rather than assumed because a validator that
rejects data an operation handles fine is worse than none — it blocks real
work and teaches people to dismiss it. An FFT needs uniform spacing;
`np.gradient` does not. A spline needs unique x; clustering reads an
observation matrix where repeated x is two ordinary observations.

Severity depends on whether a repair is coming: duplicate x is fatal to a
spline, but averaging is a defensible fix, so it is an error when nobody
will fix it and a warning when somebody will (`repairable=True`, which
`prepare_input_xy` passes).

The three failure modes this exists for all produce *wrong answers* rather
than errors — unsorted x smoothed as a sequence, a spline through duplicate
x, and `pd.to_numeric(errors="coerce")` turning a text column into an
all-NaN array with no complaint.

The check itself is `app/utils/series_validation.py`: pure numpy, no Qt, no
pandas. Tests in `app/tests/test_series_validation.py`, about half of which
assert what it must **not** reject.

#### The operations that ship

| Operation | Reads | Produces |
| --- | --- | --- |
| Fit | one series | fitted curve + parameters |
| Interpolation | one series | resampled curve |
| Smoothing | one series | smoothed curve |
| Outliers | one series | `Hide` flags on the source rows |
| Spectral | one series | spectrum on a new axis |
| Clustering | one series | cluster labels / split series |
| Statistics | one series | a report, no series |
| **Calculus** | one series | derivative or integral |
| **Peaks** | one series | located peaks + measurements |
| **Control Chart** | one series | chart values, limits, violations |
| **Function** | *nothing* | an evaluated function |

Three of these are worth knowing about before touching them.

**Calculus** builds smoothing into the derivative and baseline subtraction
into the integral, rather than leaving either as a step the user must
remember. Differentiation amplifies noise — Savitzky-Golay beats a raw
`np.gradient` by about 8x RMS on noisy data — and a peak on a raised baseline
integrates to mostly baseline: a gaussian of true area 1.77 on an offset of 5
comes out at 51.8 without subtraction.

`savgol_filter` must be given `delta` set to the sample spacing, or it returns
a derivative per *sample index* — correct only when the step happens to be 1.

**Control Chart** estimates sigma from within-subgroup variation (average
moving range over d2, or average within-subgroup range/standard deviation),
**never** from the standard deviation of all the data. That is the whole idea:
a process that has drifted has a large overall standard deviation *because* it
drifted, so limits built from it are wide enough to contain the drift and the
chart declares the process fine.

The SPC constants (d2, d3, c4, A2, D3, D4, B3, B4) are tabulated rather than
computed. c4 has a closed form, but d2 and d3 are integrals over the range
distribution with no elementary form, and using anything but the published
table would put these limits at odds with every other tool's.

X-bar limits are derived from d2 and divided by sqrt(n) rather than applying
the tabulated A2 shortcut, because A2 has the 3 of "three sigma" baked into it
and the sigma multiplier is configurable here. The two agree to ~1e-3 of the
limit, which is table rounding.

Violations carry **every** Nelson rule a point broke, not the first: the rule
numbers are historical, not a severity ranking.

**Function** is the odd one out — it reads no source series and generates one,
so it overrides `selected_series()` to return `[]`. `_run_operation` resolves
its target from the selected *axis*, which is unaffected. Its function library
comes from `FunctionScanner`, the same one the fit dialog uses, so a class
dropped into `app/functions/user_functions.py` appears in both with no
registration. Its range controls are declared in `PARAMS`; the function's own
parameters are a table, because their number and names change with the
selection and a declaration cannot express that.

Tests for all four: `app/tests/test_new_operations.py`.

### 7.3 The `.mplstyle` library

`mplstyles/` ships a large set of `.mplstyle` files, organized into
subfolders (`color/`, `color/discrete-rainbow/`, `journals/`, `languages/`,
`misc/`). The Style dropdown on the Figure panel
(`FigurePropertiesWidget._list_mplstyle_files`) walks the whole tree
(`Path.rglob("*.mplstyle")`), not just the top level, and labels each entry
`folder.subfolder.name` (dots joining the path relative to `mplstyles/`,
extension dropped) so a style's origin stays visible in a flat dropdown. The
same dropdown has a **Browse…** entry that opens a native file picker for a
style kept anywhere else entirely (`_browse_for_style_file`). Adding a style
is just dropping a `.mplstyle` file anywhere under `mplstyles/`, including a
new subfolder — no registration needed.

### 7.4 ChartPanel interaction

Everything the chart does under the pointer is wired in
`app/widgets/chart_panel.py`, through Matplotlib's own event system
rather than Qt's — Matplotlib already knows which artist owns each pixel and
can give a position in data coordinates, and redoing either against Qt
coordinates would mean reimplementing marker sizes, transforms and axis
scales for every renderer.

| Event | Does |
| --- | --- |
| `scroll_event` | Zoom about the point under the cursor |
| `pick_event` | Read out a point, a bar/wedge, or toggle a legend entry |
| `button_press_event` | Ctrl+click creates an axis annotation |
| `motion_notify_event` | Hover readout |

**Hover** is throttled to `HOVER_INTERVAL_MS` (40ms) and **blitted**. Both
matter: mouse motion arrives far faster than a hit test plus a repaint can be
done, and a full `draw()` re-runs every renderer for every series. Restoring
a cached bitmap and drawing one text box costs ~2.5ms whatever the data size,
against 12–26ms for a full draw. What grows with the data is the hit test,
which at 500k points is already the larger half of the budget.

Two traps in the blitting, both of which silently defeat it:

- The background must be captured with the annotation *hidden*, or it is
  baked in and smears across the plot.
- `_invalidate_hover_background` (on `draw_event`/`resize_event`) must drop
  only the bitmap, never `_hover_axes`. Clearing both makes every hover
  rebuild the annotation and force the full draw that blitting exists to
  avoid — the blit path is then never reached at all.
  `_discard_hover_annotation` is the separate, stronger reset, called from
  `reload()` because `figure.clear()` destroys the artist itself.

Hit distance is measured in **display pixels**, not data units: a chart of
millivolts against seconds would otherwise treat a step along x as thousands
of times nearer than one along y.

**Picking** arms lines and collections with a tolerance in points, and
patches with `picker=True` — a patch is a filled area, so "inside the shape"
is the test, and a distance from its edge leaves the middle of a tall bar
unclickable. Patches read out through `_describe_patch` rather than the point
path: a bar reports its category and value, a wedge its share. Bar
orientation comes from the `BarContainer`, not from the geometry, because a
tall thin `barh` bar and a tall thin `bar` bar are the same rectangle.

**Legend picking** toggles a series' visibility. Handles are matched to
artists by label rather than by position, so a series drawn as several
artists — a line plus its error bars — toggles as one. The entry dims instead
of disappearing, so a hidden series still has something to click. Hidden
series are skipped by the hover hit test.

Tests: `app/tests/test_chart_panel_selection.py`,
`app/tests/test_chart_panel_buffer.py`.

## 8. Styling (`app/styles/`)

- `style.py` is the shared factory: every button/menu-item/card in the app
  goes through `create_action_button` / `create_menu_item` / `create_menu` /
  `create_card_widget` rather than raw Qt constructors, so label/tooltip/icon
  come from one place (`config.json`'s `actions` catalogue,
  `action_presentation(action_id)`) and stay consistent. `MenuItem.icon`
  accepts either a string (looked up through `load_icon`: action id → SVG
  file name → Fluent glyph token, in that order) or a `QIcon` built
  elsewhere (`icon_from_svg_source`) when a menu entry has to render
  pixel-identical to an icon defined somewhere else in the app, e.g. the same
  action's own inline-SVG artwork.
- Two platform stylesheets, `macos_native.qss` and `fluent_win11.qss`, loaded
  based on `sys.platform`. They are **not** symmetric by design: the macOS
  file explicitly leaves standard controls (`QPushButton`, `QComboBox`, ...)
  unstyled because Qt's Aqua style already renders them natively, and only
  opts in bespoke, app-specific chrome by object name (read the file's own
  "PLATFORM PARITY NOTES" header before touching either file). The Windows
  file takes the opposite approach — a generic `QToolButton { ... }` fallback
  rule styles every tool button uniformly. A consequence: a new
  `QToolButton` built through `create_toolbar_button` gets Fluent styling
  "for free" on Windows but needs its own `QToolButton#<action_id>Button`
  rule added to `macos_native.qss` explicitly, or it falls through to the
  unstyled native bevel. `zoom_fitButton` (the "Adatta"/fit-to-window
  button) is the worked example.
- Icons: four backends, tried in this order by `icon_from_action_spec` (read
  its docstring for the reasoning):
  1. `SFSymbol` on macOS — the system's own set;
  2. `SegoeFluent` on Windows — likewise;
  3. `ThemeIcon` anywhere, through `QIcon.fromTheme` — a freedesktop name,
     which is what covers Linux: the desktop already has an icon theme the
     user chose, and this makes the app's Open look like every other Open on
     that machine;
  4. the SVG in `app/icons/{macOs,win11,common}/`, as the last resort.

  Inline SVG source (`icon_from_svg_source`) is separate and still the way
  plugin-carried artwork arrives.

  Adding an action means adding all four to its `config.json` entry. The
  theme name must be one Qt standardises (`QIcon.ThemeIcon`, read at import
  by `_standard_theme_icon_names`) or one listed in
  `style.EXTRA_THEME_ICON_NAMES`; a test enforces it, because the failure
  mode otherwise is silent — `fromTheme` returns a null icon on a typo and
  the SVG quietly takes over, so the only symptom is one icon that never
  looks like the rest.

  Two theme quirks are handled and worth knowing. GNOME's Adwaita dropped the
  full-colour action icons at version 45, so `document-open` is gone there
  and `document-open-symbolic` is the icon; `_theme_icon_candidates` asks for
  the plain name and then the symbolic one, which covers both Adwaita and
  Breeze. And theme icons are *not* tinted, unlike the two glyph backends:
  the theme already ships light and dark variants, and painting over one
  would replace artwork the user chose with a flat silhouette.

  The shipped SVGs are on their way out — they are drawings made once, by
  hand, that match no system in particular. `style.report_icon_sources()`
  returns the action ids each backend answers for *on the machine it runs
  on*, which is how to find out which SVGs are still doing work before
  deleting any. On a GNOME desktop it currently reports 49 of 51 from the
  theme; the two left are the Plot button and the SQL filter, which have no
  system equivalent.

## 9. Localization (`app/utils/i18n.py`)

Gettext on top of `.po`/`.mo`, but with a from-scratch `.po` reader/`.mo`
writer (no build-time dependency on gettext tooling — a `.po` edit is picked
up on next run because the `.mo` is recompiled whenever it is older). Call
sites use the English string as the message id:

```python
from app.utils.i18n import _
_("Delete…")
_("Delete {count} selected tables").format(count=n)   # dynamic content: format() the *translated* template
```

Every user-facing string that reaches a widget — labels, tooltips, menu item
text, dialog titles — must go through `_()`/`tr()`. A popup menu (or any
other UI) built from bare Python strings will simply not translate; this is
a common regression when a new menu is added by copying an existing
`MenuItem(text="...", ...)` block without noticing the sibling entries
already wrap `text` in `_()`. `app/tests/test_localization.py` and
`app/tests/test_messages.py` check catalogue coverage; a string added to the
code needs a matching `msgid`/`msgstr` pair appended to
`app/locales/it/LC_MESSAGES/datahub.po` (any position — the file is not
order-sensitive, blank-line-separated blocks — see the existing entries for
the exact format) to actually show translated, not just be translatable.

## 10. `config.json`

Machine-level preferences and the action catalogue (label/tooltip/icon per
`action_id`), read through `app/utils/config.py`'s `get_section`/`set_section`
rather than one accessor per key — see that module's docstring for why.
Anything that belongs to one figure/axis/series lives in its descriptor's
`options` instead (§3), so it travels with the `.dhub` file rather than with
the machine.

## 11. Testing

- `matplotlib.use("Agg")` is set in `app/tests/conftest.py`, so the render
  pipeline (`render_figure.py` and everything under `app/charts/`) is fully
  testable without a display — most of the suite runs this way.
- Widget tests need a `QApplication`; use the shared `qapp` session fixture
  rather than constructing one per test (Qt allows exactly one `QApplication`
  per process). Run headless with `QT_QPA_PLATFORM=offscreen`.
- **Never call `QMenu.exec()` (or any modal `exec()`) from a test** — it
  opens a real, blocking local event loop that nothing will ever click, and
  the test hangs. Split menu *construction* from *showing* it as a testable
  method — `TablePreviewPanel._build_context_menu(pos) -> QMenu | None` /
  `_show_context_menu` is the pattern — and test the builder.
- `tmp_db_path` / `test_results_dir` / `plots_dir` fixtures give each test an
  isolated `.dhub` path and a directory for saved plots
  (`DHUB_TEST_ARTIFACTS` env var to redirect).
- Run the suite: `python -m pytest app/tests -q` (add
  `QT_QPA_PLATFORM=offscreen` in a headless environment). `--run-perf`
  enables the slow performance tests, off by default.
