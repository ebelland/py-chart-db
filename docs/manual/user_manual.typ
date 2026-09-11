#set document(title: "ChartLibre — User Manual", author: "ChartLibre")
#set page(paper: "a4", margin: (x: 2.4cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

#let version = "0.1.0"

#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  v(0.4em)
  block(text(size: 20pt, weight: "bold", it.body))
  v(0.6em)
  line(length: 100%, stroke: 0.6pt + gray)
  v(0.6em)
}

#show heading.where(level: 2): it => {
  v(0.6em)
  block(text(size: 13.5pt, weight: "bold", it.body))
  v(0.2em)
}

#show heading.where(level: 3): it => {
  block(text(size: 11pt, weight: "bold", style: "italic", it.body))
}

#let note(body) = block(
  fill: rgb("#eef3fb"),
  stroke: (left: 2.5pt + rgb("#3a6ea5")),
  inset: (x: 10pt, y: 8pt),
  radius: 2pt,
  width: 100%,
)[#body]

#let code(body) = block(
  fill: rgb("#f5f5f7"),
  stroke: 0.5pt + rgb("#dcdce0"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
)[#text(font: "Menlo", size: 8.8pt, body)]

// ----------------------------------------------------------------------
// Cover
// ----------------------------------------------------------------------
#align(center)[
  #v(4cm)
  #text(size: 30pt, weight: "bold")[ChartLibre]
  #v(0.3cm)
  #text(size: 15pt, style: "italic")[User Manual]
  #v(1.5cm)
  #text(size: 11pt, fill: gray)[Application version #version]
  #v(0.3cm)
  #text(size: 10pt, fill: gray)[A desktop application for exploring data and building scientific charts]
]

#pagebreak()
#outline(title: "Contents", indent: auto)

= Introduction

ChartLibre is a desktop application for Windows, macOS and Linux that lets you import tabular data, query it with SQL, and turn it into scientific charts — from histograms to 3D surfaces — without writing code.

Every project is a single file with the extension `.dhub`: a SQLite database that holds both the imported data and the definitions of every chart (figures, axes, series and their drawing options). The file is therefore self-contained and portable — moving or sharing it carries both the data and every visualization built on top of it.

== Who this manual is for

This manual describes the application as it presents itself to a user: the main window, importing data, building queries, creating and customizing charts, running statistical operations on series, and the application's settings. A final chapter, @advanced, is aimed at developers who want to extend ChartLibre with a new chart type or a new analysis operation.

== Key concepts

- *Database (`.dhub`)*: the project file. Holds data tables and figures.
- *Table*: a set of data, either imported (from CSV, Excel, ...) or produced by a saved query.
- *Saved query*: a named SQL statement, usable anywhere a table is.
- *Figure*: one tab in the chart panel; can hold one or more axes.
- *Axis*: a single chart inside a figure, with a chart type (histogram, scatter, ...) and one or more series.
- *Series*: the data drawn on an axis, defined by a SQL query that produces the columns (*roles*) the chart type requires (e.g. `x`, `y`).
- *Renderer*: the piece of code behind a chart type — it turns a series' data into a Matplotlib drawing. See @renderers.

= Starting up and managing databases

On first launch, or whenever no file is given, ChartLibre asks which database to open:

/ New: creates an empty `.dhub` database at a path you choose.
/ Open: opens an existing `.dhub` file.
/ Load demo: loads one of the shipped demo projects and opens it immediately — useful for exploring the application without your own data.

The last database opened is remembered and offered again on the next launch.

== Saving and duplicating a project

A SQLite database writes its own changes straight to disk on every operation, so there is no separate "Save" for data. Two related commands are available instead:

/ *Save As...*: saves the current database under a new name/path and switches to working on that copy, leaving the original untouched.
/ *Optimize DB*: checks the database for problems, reports them, and compacts it (`VACUUM`, `ANALYZE`) to shrink it on disk and speed up opening it.

= The main window

#figure(
  image("screenshot_main_window.png", width: 100%),
  caption: [The main window: table list and data preview on the left, chart panel on the right.],
)

The main window splits into two areas.

== Navigation rail (left)

A vertical column of icons switches between the application's main sections:

- *Data* — the list of tables and queries in the database, with a data preview.
- *Chart Options* — properties of the currently selected figure, axis and series.
- *Series Operations* — *Plot*, plus the analysis tools (fit, statistics, filtering, ...) that apply to a series' data.
- *Menu* — opens the application's main menu (new, open, import, settings, credits, ...).

== Data panel

Lists the tables in the database (saved queries are marked with a *Q* icon) and, once one is selected, shows a preview with its first rows and columns. This is also where importing data and opening the Query Builder start.

== Chart panel

Occupies the right side of the window and is organized into tabs — each tab is a *figure*. The toolbar above the chart offers Matplotlib's usual navigation tools:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Home*], [Return the chart to its initial view],
  [*Arrows ←/→*], [Step back/forward through the zoom history],
  [*Pan*], [Drag to move the visible area],
  [*Zoom*], [Draw a rectangle to zoom into an area],
)

*Zoom to fit* brings the whole chart back within the panel's bounds in one click.

The *Plot* button, at the top of the Series Operations panel, creates a new figure, configured as described in @creating-a-chart.

= Importing data

*Import* opens a dialog with four ways to bring data in:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Open*], [A local file: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xlsm`, `.xls`, `.json`, `.xml`.],
  [*Paste*], [Whatever tabular text is currently on the clipboard.],
  [*Database*], [One table — or the result of a query — read out of another database: SQLite (including another `.dhub` file), PostgreSQL, or MySQL.],
  [*Web*], [Whatever an http(s) URL returns — a CSV export, a JSON API, a spreadsheet.],
)

Whichever way the data arrives, the same preview, column-type mapping and destination-table name apply before anything is written. For Excel sources with more than one sheet, the dialog asks which one to import.

The dialog's left panel is grouped into *Source* (where the data comes from, including the web row), *Read options* (destination table name, header row, skipped rows, delimiter, encoding) and *Columns* (the per-column type mapping). The read options describe how to parse *text*: when the source is a database table they are disabled, because a table already has its own column types, no delimiter and no header row to detect.

*Database* opens its own small window:

+ Pick an engine — a SQLite file, another ChartLibre project (`.dhub`), PostgreSQL or MySQL.
+ Fill in a file path, or a host/port/username/password. For a server, *Connect* asks it which databases exist and offers them in a list, so the name does not have to be typed from memory; picking one lists its tables.
+ Choose a table — never any internal bookkeeping tables ChartLibre itself may have added, for a SQLite/`.dhub` source — or tick *Use a query* and write a `SELECT` instead, for a join, a filter or an aggregate.
+ Confirm to bring the result back to the import dialog like any other source.

The last connection is remembered, password excepted, so reopening the window lands on the same server and table.

*Web* has its own row: a quick-pick menu of ready-made public datasets grouped by subject, the URL itself, and *Fetch*. Picking an entry fills the URL in rather than downloading immediately, so it can be read and edited first. *Add source* saves a URL of your own to that menu under a name you choose, and *Delete source* removes one you added — the entries that ship with the application cannot be deleted.

#note[
  An imported table stays available to every query and chart in the project: importing is a one-time read, not a live link back to the original file, database or URL — the data does not change on its own afterwards. Every source *except* a paste can be re-read later, though: right-click the table and choose *Update link* to replace its contents with a fresh read from the same file, database table, query or URL.
]

#note[
  A PostgreSQL or MySQL connection's password is never saved — not in the project file, not anywhere on disk. *Update link* asks for it again each time, because a `.dhub` project is exactly the kind of file that gets copied, emailed or committed without a second thought, and a password sitting in plain JSON inside it would travel right along.
]

#note[
  *Web* only fetches `http://` and `https://` URLs — never a local path — so pasting a link here cannot read a file off disk by way of a `file://` address.
]

== Exporting data

The content of a table or the result of a query can be exported with:

/ *Export CSV*: writes the data to a `.csv` text file.
/ *Export Excel*: writes the data to an `.xlsx` workbook.

= Query Builder

The *Query Builder* (menu *Query Builder*) writes, checks and saves SQL statements that can be used as tables — handy for filtering, joining or aggregating imported data without duplicating it.

The editor has a *Run* button to check the result before saving, and a set of ready-made snippets that insert the right SQL skeleton:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Select*], [`SELECT` over a chosen table],
  [*Filter*], [`WHERE` with the comparisons spelled out],
  [*Join*], [`LEFT JOIN` keeping every row of the chosen table],
  [*Order*], [`ORDER BY` with an explicit direction],
  [*Summary*], [Count, average, min and max, grouped by a column],
  [*Union*], [`UNION ALL` of two tables with the same columns],
)

A saved query appears in the table list with a *Q* icon and can be used as the data source for any series, exactly like an imported table. Unlike a table, its content is computed on the fly rather than stored: if the source data changes, the query returns fresh results the next time it runs.

= Creating a chart <creating-a-chart>

*Plot*, at the top of the Series Operations panel, opens the chart-creation window: pick a *chart type* (a renderer) and define the first series — the SQL query that supplies the data.

Every chart type requires the query to produce columns with specific names, its *roles* — for instance `x` and `y` for a scatter plot. The window shows the roles a type requires and a link to that type's Matplotlib documentation.

== Chart types <renderers>

ChartLibre ships 29 chart types (*renderers*), grouped into the same categories Matplotlib's own documentation uses. Each is a self-contained piece of code that receives the rows a series' SQL query returns and draws them; see @advanced-renderer for how a new one is added.

The chart picker lists them section by section and its search box matches a name or a whole category, so typing "stat" brings up the statistical family at once.

=== Pairwise data (x, y)

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Bar Chart*], [Vertical bar chart],
  [*Horizontal Bar Chart*], [Horizontal bar chart],
  [*Broken Bar*], [Interval bars grouped by category (Gantt-style)],
  [*Broken Bar (Vertical)*], [The same, stacked in columns],
  [*Scatter Plot*], [Scatter plot; optional `color` and `size` columns drive a colour map and marker sizing],
  [*Fill Between*], [The region between two y curves over a shared x — confidence bands, tolerance limits],
  [*Stack Plot*], [Series stacked into filled bands, showing how a total divides into its parts],
  [*Stem Plot*], [A stem from a baseline to each value — impulses, spectra, anything sampled at discrete x],
  [*Stairs*], [A step outline over bin edges, for values that hold constant between them],
  [*Table*], [A data table, rows and columns of text rather than a plot],
  [*Text*], [Text labels at data points, spread apart to avoid overlap],
  [*Time Series*], [Time series, with numeric or timestamp x],
  [*Timeline*], [Dated events on a baseline, each on its own stem — release histories, event lists],
)

=== Statistical distributions

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Histogram*], [Histogram, supports multiple datasets],
  [*Box Plot*], [Box-and-whisker plot],
  [*Violin Plot*], [Estimated distribution of one or more samples],
  [*ECDF*], [Empirical cumulative distribution function],
  [*Pareto Chart*], [Categories sorted by descending magnitude with a cumulative-percentage line],
  [*Pie Chart*], [Pie or donut chart of one series],
)

=== Gridded data

Values sampled on a complete, evenly spaced x/y grid.

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Contour Plot*], [Filled or line contours of z over a regular x/y grid],
  [*Quiver*], [An arrow per sample showing a vector field's direction and magnitude (u/v components)],
  [*Stream Plot*], [Streamlines traced through a vector field — where a flow goes, rather than what it does at each sample],
  [*Wind Barbs*], [A barb per point encoding speed in flags, readable where a field of arrows is not],
)

=== Irregularly gridded data

The same quantities measured wherever they could be measured, with no grid to assume.

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Contour Plot (Scattered)*], [Filled or line contours for scattered x/y/z data],
  [*Triangular Mesh*], [The triangulation itself, as edges and vertices — shows what an interpolating chart will assume between samples],
  [*Triangular Color Mesh*], [Scattered samples triangulated and filled by value],
)

=== 3D and volumetric data

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [*Surface Plot*], [3D surface over a regular x/y grid],
  [*Surface Plot (Scattered)*], [3D triangulated surface for scattered (non-gridded) x/y/z data],
  [*Scatter Plot (3D)*], [Points in three dimensions, optionally coloured and sized by further columns],
)

#note[
  A chart type that needs a complete grid says so and draws nothing when the rows it is given are not one — it never interpolates the missing cells and calls the result a measurement. Its scattered counterpart is named in the log entry; switch the axis to that one instead.
]

== How a renderer reads a series

A renderer never sees your raw table — it sees the result of the series' SQL query, aliased to its required *roles*. A Scatter Plot needs `x` and `y`, so its series query is written as:

#code[```
SELECT pressure AS x, flow AS y FROM operating_points
```]

Optional roles work the same way and unlock extra behaviour when present — for instance `color` and `size` on a Scatter Plot drive a colour map and marker sizing, and `yerr`/`yerr_low`/`yerr_high` add error bars. The chart-creation window and the Chart Options panel both list which roles a given type accepts.

Every renderer also splits its adjustable settings into two families, both editable from Chart Options:

- *Plot arguments* (forwarded to Matplotlib as-is — line width, marker, transparency, colormap, ...).
- *Options* (consumed by the renderer itself and never forwarded to Matplotlib — whether to draw a legend, how many bins, which fit to overlay, ...).

== Adding more series or axes

A figure can hold several axes (each drawn side by side or overlaid within the same tab), and each axis can hold several series, up to the maximum the chart type allows — some types, like Pie Chart, accept only one series; the application then proposes a new axis instead of a second series that would be dropped.

*Add series* in the Chart Options panel adds a new series to the selected axis.

== Customizing a chart

From Chart Options, depending on the selected level:

- *Figure*: title, layout of its axes.
- *Axis*: title, axis labels, grid, legend, and the chart type's own options (e.g. the number of bins for a histogram).
- *Series*: the source query, colour, line or marker style, and any Matplotlib arguments forwarded straight to the drawing call (line width, transparency, ...).

Every setting shows a contextual description, so reading Matplotlib's own documentation is rarely necessary to understand what a parameter does.

= Series operations <series-operations>

*Series Operations* applies statistical or mathematical transformations to an existing series' data, previewing the result before it is committed to the chart. Every operation dialog shares the same layout: on the left, the source axis/series, a model and its parameters; on the right, a preview of the result and a log of the computation.

The panel itself is a grid of square buttons grouped by what they are for — *Plot*, *Analysis*, *Statistics*, *Signal Processing*, *Modeling* — with a bar along the bottom that describes whichever button the pointer (or the keyboard focus) is on.

#table(
  columns: (auto, 1fr, 1.4fr),
  stroke: none,
  inset: 6pt,
  [*Peaks*], [Find and measure peaks], [Detects peaks in a series and reports their position, height and width.],
  [*Roots*], [Find where a series crosses a level], [Locates the x values at which a series crosses a chosen level — zero by default — by interpolating between the samples either side.],
  [*Calculus*], [Differentiate or integrate], [Computes the numerical derivative or the cumulative integral of a series, with smoothing built into the first and baseline subtraction into the second.],
  [*Statistics*], [Compute metrics], [Reports summary statistics (mean, standard deviation, quantiles, ...) for a series.],
  [*Outliers*], [Detect anomalies], [Flags points that deviate from the rest of a series by a chosen statistical criterion.],
  [*Clustering*], [Group similar data], [Groups a series' points into clusters (k-means, DBSCAN, ...) and labels each point by cluster.],
  [*Control Chart*], [Monitor process stability], [Builds a statistical control chart and flags rule violations — I-MR, X-bar-R and X-bar-S for measurements, p, np, c and u for counts.],
  [*Smoothing*], [Reduce noise], [Applies a smoothing model (moving average, Savitzky-Golay, ...) to reduce noise while preserving the underlying shape.],
  [*Spectral Analysis*], [Analyse frequencies], [Power spectral density, cross-spectral density, coherence, magnitude/phase spectra and auto/cross-correlation, on their own new axis.],
  [*Filtering*], [Filter, detrend or demodulate], [Low/high/band-pass and band-stop filtering (Butterworth, Chebyshev, Bessel, FIR), trend removal, and the Hilbert envelope of a signal.],
  [*Baseline Correction*], [Subtract a background], [Removes a drifting background from a spectrum — asymmetric least squares or a rubber band — and keeps the baseline it removed as its own series.],
  [*Fit*], [Fit data models], [Fits a mathematical model (Gaussian, exponential, polynomial, a user function, ...) to a series by least squares, and adds the fitted curve alongside the data. Optionally adds a residuals chart and a measured-vs-fit chart.],
  [*Interpolation*], [Fill missing values], [Fills gaps in a series (linear, spline, nearest, ...), producing a complete curve from a sparse one.],
  [*Function*], [Plot a function], [Evaluates a function over a range and plots it — the one operation that reads no source series at all.],
)

A typical workflow is: select the axis and series to operate on, choose a model and its parameters, click *Preview* to see the result superimposed on the chart, adjust the parameters if needed, then confirm to add the result as a new series (or table) permanently. See @advanced-operation for how a new operation is added.

#note[
  An operation reads *one* source series. If several are ticked, the first one is used and the others are ignored — the status bar says which one it took. To operate on a different series, untick the others, or move the one you want to the top of the list.
]

= Appearance and styles

== Matplotlib styles

*Edit Matplotlib styles* lets you adjust the base graphical parameters (rcParams) that govern the look of every chart — default colours, fonts, line widths. From this window you can:

- Load an existing `.mplstyle` file into the editor.
- Add an rcParam property to the editor.
- Reset a property to its Matplotlib default.
- Save the changes as a new `.mplstyle` file.

== Application settings

*Settings* gathers the general preferences:

/ *App style*: the look of the interface controls. Includes the platform's native styles plus, when installed, extra Qt styles (e.g. Breeze, Oxygen, QtCurve).
/ *Language*: the interface language. Besides Auto (which follows the operating system's language), Italian is available.

= Application log

The *Log viewer* shows the history of the application's internal operations (startup, opening a database, errors) — useful for diagnosing unexpected behaviour or attaching details to a bug report.

= Demo projects

*Load demo* opens a pre-filled `.dhub` database with sample tables and a set of figures already configured across several chart types. It is the fastest way to explore the application's features without preparing your own data, and a good place to copy a series' or an axis' settings from into a real project. It loads straight into your home directory under the demo's own name — no save dialog to answer first — and loading the same demo again simply replaces that file with a fresh copy, so an edited demo is never mistaken for your own work.

= Credits

*Credits* lists the application, the version in use, and the open-source libraries it is built on (including Qt/PySide6, Matplotlib, NumPy, pandas, SciPy and statsmodels).

= Advanced: extending ChartLibre <advanced>

ChartLibre discovers chart types and series operations the same way: by scanning a folder for Python classes that directly subclass a known base class, at import time — no registration list to edit and keep in sync. Dropping a well-formed file into the right folder is enough for it to appear in the application.

== Writing a custom chart renderer <advanced-renderer>

A chart type lives entirely in one file under `app/charts/`, as a class that subclasses `BaseAxisRenderer` (from `app.charts.base`) *directly* — the discovery scanner in `app/scanners/axis_renderer_scanner.py` matches that exact base-class name in the source, even when the class also inherits behaviour from another renderer.

The class is described entirely through its own attributes:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [`Name`], [Display name, and the `chart_type` value stored in the database. Renaming it orphans existing axes unless an alias is added to `CHART_TYPE_ALIASES` in the scanner.],
  [`Category`], [Which family of plot this is, following Matplotlib's own taxonomy (e.g. "Pairwise data", "Statistical distributions").],
  [`Description` / `Link`], [Shown in the chart picker and in the axis properties panel.],
  [`RequiredRoles` / `OptionalRoles`], [Column names the series' SQL query must (or may) produce — see @renderers.],
  [`Kwargs`], [Matplotlib keyword arguments forwarded verbatim to the plot call, as `{name: metadata}`. The metadata (`default`, `type`, `min`/`max`, `kind`, `group`, `description`) drives the generated editor UI automatically.],
  [`Options`], [Settings the renderer consumes itself and never forwards to Matplotlib — same metadata shape, opposite destination.],
  [`MaxSeries`], [How many series this renderer can draw on one axis, or `None` for any number.],
)

The class then implements one method:

#code[```
def render_axis(self, ax, series: list[SeriesData], options: dict | None = None) -> None:
    ...
```]

which receives one `SeriesData` per series — already queried, with its columns in `sd.df` — and draws them onto `ax` (a Matplotlib `Axes`).

`sd.df` is a `SeriesFrame` (`app.data.series_frame`): columns of NumPy arrays read straight from SQLite, read the way a DataFrame is (`sd.df["x"]`, `"x" in sd.df.columns`, `sd.df.loc[mask, "color"]`), with `sd.df.to_pandas()` available for the rare operation that genuinely needs a DataFrame. The developer guide's §4.1 covers it.

*Practical steps:*

+ Create `app/charts/my_chart.py`.
+ Import `BaseAxisRenderer` and, if useful, `SeriesData` from `app.charts.base`.
+ Define a class, e.g. `MyChartAxisRenderer(BaseAxisRenderer)`, setting `Name`, `Category`, `Description`, `RequiredRoles`/`OptionalRoles`, and optionally `Kwargs`/`Options`.
+ Implement `render_axis` using ordinary Matplotlib calls on `ax`.
+ Restart the application (or reopen the chart-creation window) — the new type appears in the picker with no further wiring.

`app/charts/text.py` is a good, compact, complete example to read or copy as a starting point: a handful of roles, a short `Kwargs` block, and a straightforward `render_axis`.

== Writing a custom series operation <advanced-operation>

A series operation lives in `app/series_operations/`, as a class that subclasses `SeriesOperationDialogBase` (from `app.series_operations.dialog_base`) *directly* — discovered the same way, by `app/scanners/series_operation_scanner.py`.

The base class supplies the whole dialog shell (series picker, parameter form, preview pane, action buttons) and the plumbing that turns a result into a new series or table on the chart. A subclass sets a few class attributes —

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [`Name`], [Display name shown in the Series Operations list.],
  [`Description`], [One-line summary shown next to the name.],
  [`Icon`], [An inline SVG source string for the operation's icon.],
)

— and overrides the hooks the base class calls at the right moments. The ones every operation implements are:

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: 6pt,
  [`build_parameter_selector()`], [Build the parameter form. Simple operations describe their parameters declaratively with the helpers in `app.series_operations.parameter_spec` (`FloatParam`, `IntParam`, `ChoiceParam`, ...) rather than laying out widgets by hand.],
  [`compute_results()`], [Run the actual computation over the selected series and return its results.],
  [`result_to_frame()` / `result_series_spec(s)()`], [Turn a result into the DataFrame and series metadata (name, roles, style) the base class writes to the chart.],
  [`format_results()`], [Render the results as the text/HTML shown in the preview pane.],
  [`refresh_results()`], [Recompute and redraw the preview after a parameter changes.],
)

*Practical steps:*

+ Create `app/series_operations/my_operation_dialog.py`.
+ Subclass `SeriesOperationDialogBase`, set `Name`, `Description`, `Icon`.
+ Declare parameters (e.g. with `FloatParam`/`IntParam`/`ChoiceParam`) and implement `build_parameter_selector`.
+ Implement `compute_results`, `result_to_frame`/`result_series_spec`, and `format_results`.
+ Restart the application — the operation appears in the Series Operations list.

`app/series_operations/function_dialog.py` is the smallest complete dialog and a reasonable starting template; `app/series_operations/calculus_dialog.py` is a good second read for an operation that consumes an existing series rather than generating one from scratch.
