"""Dialog for outlier detection and Hide-column marking on chart series.

The dialog only owns UI, preview and numeric outlier detection.  All database
management is delegated to SqliteRepo:
- creating/resetting the Hide column
- marking rows as hidden
- updating series SQL with the Hide filter
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QFormLayout,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import median_filter

from app.data.data_source import parse_roles, quote_identifier, row_value
from app.data.sqlite_repo import SqliteRepo
from app.series_operations.parameter_spec import FloatParam, IntParam
from app.series_operations.series_operation_dialog_base import (
    ResultSeriesSpec,
    SeriesOperationDialogBase,
    generated_table_name,
)
from app.logs.logger import applogger
from app.utils.coercion import to_numeric_axis

# Fewer than this and every estimator here is meaningless: a median absolute
# deviation over two points says nothing about either of them.
MIN_POINTS: int = 3
from app.utils.messages import show_message
from app.styles.style import (
    create_doc_link,
    set_doc_link,
)
from app.utils.i18n import _


OUTLIER_ZSCORE = "Z-score threshold"
OUTLIER_IQR = "Interquartile range"
OUTLIER_MAD = "Median absolute deviation"
OUTLIER_ROLLING = "Rolling median residual"

OUTLIER_METHODS = (
    OUTLIER_ZSCORE,
    OUTLIER_IQR,
    OUTLIER_MAD,
    OUTLIER_ROLLING,
)

OUTLIER_DOCS = {
    OUTLIER_ZSCORE: (
        "Z-score outlier detection",
        "https://en.wikipedia.org/wiki/Standard_score",
    ),
    OUTLIER_IQR: (
        "IQR outlier detection",
        "https://en.wikipedia.org/wiki/Interquartile_range",
    ),
    OUTLIER_MAD: (
        "Median absolute deviation",
        "https://en.wikipedia.org/wiki/Median_absolute_deviation",
    ),
    OUTLIER_ROLLING: (
        "Rolling median residuals",
        "https://en.wikipedia.org/wiki/Moving_average#Median_filter",
    ),
}

@dataclass(slots=True)
class OutlierResult:
    """Outlier detection result for one source series."""

    source_name: str
    result_name: str
    model: str
    source_table: str
    x_col: str
    y_col: str
    x: np.ndarray
    y: np.ndarray
    outlier_x: np.ndarray
    outlier_y: np.ndarray
    outlier_rowids: np.ndarray
    metadata: dict[str, Any]
    outlier_count: int
    message: str

    def to_frame(self) -> pd.DataFrame:
        """Return a small preview frame; not used for persistence."""
        return pd.DataFrame(
            {
                "source_name": self.source_name,
                "model": self.model,
                "x": self.x,
                "y": self.y,
                "hide": False,
            }
        )

class SeriesOutlierDialog(SeriesOperationDialogBase):
    """Dialog to detect outliers and mark source rows with Hide=True."""
    Name: str = "Outliers"
    Description = "Detect anomalies"

    # The rolling-median detectors walk the series in order; an unsorted x
    # makes every window span an arbitrary set of points, so the residuals
    # it flags are not the outliers.
    INPUT_REQUIRES_SORTED_X = True
    INPUT_MINIMUM_POINTS = 3

    # Replaces the three hand-built spin boxes, their three signal
    # connections, and _refresh_visibility's three set_row_visible calls. The
    # visibility rules that used to live in code are now the same data the
    # widgets are built from, so the two cannot drift apart.
    PARAMS = (
        FloatParam(
            "threshold",
            "Threshold:",
            tooltip="Threshold multiplier applied to the method's spread estimate.",
            default_value=3.0,
            minimum=0.1,
            maximum=30.0,
            visible_for={"model": (OUTLIER_ZSCORE, OUTLIER_MAD, OUTLIER_ROLLING)},
        ),
        FloatParam(
            "iqr_factor",
            "IQR multiplier:",
            tooltip="Multiplier applied to IQR outlier bounds.",
            default_value=1.5,
            minimum=0.5,
            maximum=10.0,
            visible_for={"model": (OUTLIER_IQR,)},
        ),
        IntParam(
            "window",
            "Window size:",
            tooltip="Rolling window size, in points, used to compute the local median.",
            default_value=11,
            minimum=3,
            maximum=9999,
            odd_only=True,
            visible_for={"model": (OUTLIER_ROLLING,)},
        ),
    )

    Icon = """
    <circle cx="7" cy="8" r="1.3"/>
    <circle cx="10.5" cy="11" r="1.3"/>
    <circle cx="8.5" cy="15" r="1.3"/>
    <circle cx="14" cy="9" r="1.3"/>
    <circle cx="16" cy="14" r="1.3"/>
    <circle cx="19" cy="5.5" r="1.5"/>
    <path d="M17.8 6.7l-2 2"/>
    """
    def __init__(self, *, repo: SqliteRepo, figure_id: int, parent: QWidget | None = None) -> None:
        if repo is None:
            applogger.error("SeriesOutlierDialog requires a repository instance.")

        self._last_results: list[OutlierResult] = []
        self._parameter_form: QFormLayout | None = None
        self._preview_active = False
        self._preview_hide_snapshots: dict[str, list[int]] = {}
        self._preview_series_sql: dict[int, str] = {}
        self._preview_state_tables: set[str] = set()

        super().__init__(
            repo=repo,
            figure_id=figure_id,
            title="Series Outliers",
            parent=parent,
            width=720,
            height=640,
        )
        self.setModal(True)
        self.series_selector.set_series_filter(self._has_query)
        self._populate_axes()
        self._refresh_methods()
        self._refresh_visibility()
        self.refresh_results()

    def init_operation_widgets(self) -> None:
        self._doc_link = create_doc_link(self)
        self._parameter_form = None

    def build_model_selector(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        form_container = QWidget(panel)
        form_layout = QFormLayout(form_container)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo.addItems(OUTLIER_METHODS)
        self.model_combo.setToolTip(_("Choose the outlier detection method."))
        form_layout.addRow(_("Model:"), self.model_combo)

        form_layout.addRow(_("Docs:"), self._doc_link)

        layout.addWidget(form_container)
        return panel

    def connect_operation_signals(self) -> None:
        # Only the model combo: ParameterForm connects every declared
        # parameter to refresh_results when it builds them.
        self.model_combo.currentIndexChanged.connect(self._refresh_visibility)
        self.model_combo.currentIndexChanged.connect(self.refresh_results)

    def _populate_axes(self) -> None:
        self.series_selector.reload(select_all_series=True)

    def _refresh_methods(self) -> None:
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(OUTLIER_METHODS)
        index = self.model_combo.findText(current)
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)
        self.model_combo.blockSignals(False)
        self._refresh_visibility()

    def _refresh_visibility(self) -> None:
        """Re-evaluate the declared visibility rules, and update the doc link.

        The rows are handled by the parameter form from the ``visible_for``
        declarations; only the documentation link, which is not a parameter,
        is still set here.
        """
        model = self.model_combo.currentText()
        form = getattr(self, "_parameter_form_spec", None)
        if form is not None:
            form.refresh_visibility()
        title, url = OUTLIER_DOCS[model]
        set_doc_link(self._doc_link, title, url)

    def refresh_results(self) -> None:
        try:
            results = self.compute_results()
        except Exception as exc:
            self._last_results = []
            self.set_results_text(f"Error:\n{exc}")
            return

        self._last_results = results
        self.set_results_text(self.format_results(results) if results else "Select one or more source series.")

    def compute_results(self) -> list[OutlierResult]:
        """Detect outliers for each selected source series.

        No selection is a normal dialog startup state, so return an empty list
        instead of raising and showing an error in the preview pane.
        """
        selected = self.selected_series()
        if not selected:
            return []

        usable = self._report_series_without_xy(selected)
        if not usable:
            return []

        model = self.model_combo.currentText()
        params = self._params()
        results: list[OutlierResult] = []
        errors: list[str] = []

        for row in usable:
            try:
                results.append(self._detect_outliers(row, model, params))
            except Exception as exc:
                errors.append(f"{self._series_display_name(row)}: {exc}")

        if errors and not results:
            applogger.error("\n".join(errors))
        if errors:
            show_message(
                self,
                "series.some_failed",
                title=self.operation_label,
                errors="\n".join(errors),
            )
        return results

    def _has_query(self, row: Any) -> bool:
        """Only expose source series backed by an SQL query."""
        return bool(row["sql_query"] != "")

    @staticmethod
    def _missing_xy_roles(row: Any) -> list[str]:
        """Return the x/y roles this series does not define.

        Outlier detection is a 1D operation over y ordered by x, and the rows it
        hides are identified by rowid in the source table, so both roles are
        structurally required - there is no sensible default for either.
        """
        roles = parse_roles(row_value(row, "roles"))
        return [
            role
            for role in ("x", "y")
            if not str(roles.get(role, "") or "").strip()
        ]

    def _report_series_without_xy(self, rows: Sequence[Any]) -> list[Any]:
        """Return only the series that can be processed, warning about the rest.

        Two reasons a series is rejected, and the user is told which.

        Without x and y the SQL built for detection is malformed, so the
        operation used to fail with a database error that said nothing about
        the actual cause.

        A series over a saved query has no table to write to: hiding an outlier
        sets a Hide flag on a real row, found by rowid, and a query result has
        neither.  The operation cannot be made to work there - the query would
        have to be materialised first - so it says so instead of failing later
        with "no such column: Hide".
        """
        usable: list[Any] = []
        rejected: list[str] = []
        query_backed: list[str] = []

        for row in rows:
            if not self._repo.is_table_backed_sql(str(row["sql_query"])):
                query_backed.append(f"• {self._series_display_name(row)}")
                continue

            missing = self._missing_xy_roles(row)
            if missing:
                rejected.append(
                    f"• {self._series_display_name(row)} "
                    f"(missing role{'s' if len(missing) > 1 else ''}: {', '.join(missing)})"
                )
            else:
                usable.append(row)

        if query_backed:
            show_message(
                self,
                "series.outliers_need_a_table",
                title=self.operation_label,
                series="\n".join(query_backed),
            )

        if rejected:
            show_message(
                self,
                "series.roles_missing",
                title=self.operation_label,
                series="\n".join(rejected),
            )

        return usable

    @staticmethod
    def _source_column_for_alias(sql_query: str, alias: str) -> str:
        """Return the source column projected as ``alias`` in a simple SELECT.

        Some stored roles contain the renderer aliases ``x``/``y`` rather than
        the physical table column names. When we build a table-writeback query
        from those roles, SQLite can return the literal strings 'x' and 'y' if
        no physical columns named x/y exist. This resolver maps aliases back to
        the real source fields from expressions like ``hour AS x``.
        """
        alias_text = str(alias or "").strip()
        if not alias_text:
            return ""
        quoted_alias = re.escape(alias_text)
        alias_pattern = (
            r'"' + quoted_alias + r'"'
            r'|\[' + quoted_alias + r'\]'
            r'|`' + quoted_alias + r'`'
            r'|' + quoted_alias + r'(?=\s|,|$)'
        )
        identifier = (
            r'"([^"\\]*(?:\\.[^"\\]*)*)"'
            r'|\[([^\]]+)\]'
            r'|`([^`]+)`'
            r'|([A-Za-z_][A-Za-z0-9_]*)'
        )
        pattern = identifier + r'\s+AS\s+(?:' + alias_pattern + r')'
        match = re.search(pattern, str(sql_query), flags=re.IGNORECASE)
        if match is None:
            return ""
        for group in match.groups():
            if group:
                return str(group)
        return ""

    @staticmethod
    def _literal_alias_column(series: pd.Series, alias: str) -> bool:
        """True when a column consists only of the alias text, e.g. 'x'."""
        values = series.dropna().astype(str).str.strip().head(20)
        return not values.empty and bool((values == str(alias)).all())

    def _load_outlier_source_frame(self, choice: Any, roles: Mapping[str, Any]) -> pd.DataFrame:
        """Return source rows with normalized x/y and __rowid__ columns.

        The repository helper is preferred because it knows how to expose rowid
        for Hide updates. If it returns literal alias values (for example x='x',
        y='y'), resolve the renderer aliases back to the source columns from the
        SELECT list and reload directly from the series SQL.
        """
        frame = self._repo.query_series_frame_for_hide(
            sql_query=choice["sql_query"],
            roles=roles,
        )
        if self._frame_has_usable_xy(frame):
            return frame

        try:
            direct = self._repo.query_df(str(choice["sql_query"]))
        except Exception:
            return frame

        if direct.empty:
            return frame

        sql_query = str(choice["sql_query"])
        x_role = str(roles.get("x", "x") or "x").strip()
        y_role = str(roles.get("y", "y") or "y").strip()

        # Role values are often renderer aliases. Prefer actual role columns
        # when they exist and are not literal alias text; otherwise resolve the
        # alias to its physical source projection from the SELECT list.
        x_col = x_role if x_role in direct.columns else ""
        y_col = y_role if y_role in direct.columns else ""

        if x_col and self._literal_alias_column(direct[x_col], x_role):
            x_col = ""
        if y_col and self._literal_alias_column(direct[y_col], y_role):
            y_col = ""

        if not x_col:
            resolved = self._source_column_for_alias(sql_query, x_role)
            x_col = resolved if resolved in direct.columns else ("x" if "x" in direct.columns else "")
        if not y_col:
            resolved = self._source_column_for_alias(sql_query, y_role)
            y_col = resolved if resolved in direct.columns else ("y" if "y" in direct.columns else "")

        if not x_col or not y_col:
            return frame

        output = pd.DataFrame({"x": direct[x_col], "y": direct[y_col]})
        if "__rowid__" in direct.columns:
            output["__rowid__"] = direct["__rowid__"]
        elif "rowid" in direct.columns:
            output["__rowid__"] = direct["rowid"]
        elif "_rowid_" in direct.columns:
            output["__rowid__"] = direct["_rowid_"]
        else:
            # Last resort: ask the repo helper for rowids but replace only x/y.
            # This is safe when both frames come from the same ordered series SQL.
            if "__rowid__" not in frame.columns or len(frame) != len(output):
                return frame
            output["__rowid__"] = frame["__rowid__"].to_numpy()

        if self._frame_has_usable_xy(output):
            return output
        return frame

    @staticmethod
    def _frame_has_usable_xy(frame: pd.DataFrame) -> bool:
        """True when a frame contains at least MIN_POINTS finite x/y pairs."""
        if frame.empty or "x" not in frame.columns or "y" not in frame.columns:
            return False
        try:
            raw_x = to_numeric_axis(frame["x"])
            raw_y = to_numeric_axis(frame["y"])
        except Exception:
            return False
        return int(np.count_nonzero(np.isfinite(raw_x) & np.isfinite(raw_y))) >= MIN_POINTS

    @staticmethod
    def _column_diagnostics(frame: pd.DataFrame) -> str:
        """Return compact x/y dtype/sample diagnostics for failure messages."""
        parts: list[str] = []
        for col in ("x", "y"):
            if col not in frame.columns:
                parts.append(f"{col}=<missing>")
                continue
            series = frame[col]
            samples = [str(v) for v in series.dropna().head(3).tolist()]
            parts.append(f"{col} dtype={series.dtype}, sample={samples}")
        return "; ".join(parts)

    def _detect_outliers(
        self,
        choice: Any,
        model: str,
        params: Mapping[str, Any],
    ) -> OutlierResult:
        """Load one source series, compute the outlier mask, and map rowids."""
        roles = parse_roles(choice["roles"])
        source_table = self._repo.query_source_table(choice["sql_query"])
        source_df = self._load_outlier_source_frame(choice, roles)

        # to_numeric alone would turn a timestamp column into all-NaN, and the
        # only symptom would be this method reporting "not enough points"
        # about a table with a million rows.  See utils.coercion.
        raw_x = to_numeric_axis(source_df["x"])
        raw_y = to_numeric_axis(source_df["y"])
        raw_rowids = source_df["__rowid__"].to_numpy(dtype=int)

        # Report only - never prepare_input_xy here. The mask this method
        # computes is mapped back to source rows through raw_rowids, so sorting
        # or merging x would move each mark onto a different row. Unsorted x is
        # a real problem for the rolling detectors, but the fix belongs in the
        # source data, and the warning says so.
        self.validate_input_xy(
            raw_x, raw_y, label=str(choice["name"]), raise_on_error=False
        )

        finite_mask = np.isfinite(raw_x) & np.isfinite(raw_y)
        finite_count = int(np.count_nonzero(finite_mask))
        if finite_count < MIN_POINTS:
            # Say what was actually found: "3 points required" against a full
            # table sends the user looking in the wrong place.
            diagnostics = self._column_diagnostics(source_df)
            applogger.error(
                "%s: %d row(s) read, %d usable X/Y pair(s), %d required. "
                "Rows already hidden are excluded. Check that roles map to the "
                "actual source columns and that X/Y are numeric or date-like. %s",
                self._series_display_name(choice),
                len(source_df),
                finite_count,
                MIN_POINTS,
                diagnostics,
            )
            raise ValueError(
                f"{self._series_display_name(choice)} has {finite_count} usable X/Y pair(s); "
                f"{MIN_POINTS} required. {diagnostics}"
            )

        x_values = raw_x[finite_mask]
        y_values = raw_y[finite_mask]
        rowids = raw_rowids[finite_mask]

        order = np.argsort(x_values)
        x_sorted = x_values[order]
        y_sorted = y_values[order]
        rowids_sorted = rowids[order]
        mask_sorted = self._outlier_mask(y_sorted, model, params)
        outlier_count = int(np.count_nonzero(mask_sorted))
        outlier_rowids = rowids_sorted[mask_sorted]

        x_out = x_sorted[~mask_sorted]
        y_out = y_sorted[~mask_sorted]
        message = f"Detected {outlier_count} outlier(s); apply marks source rows Hide=True"

        return OutlierResult(
            source_name=choice["name"],
            result_name=f"{choice['name']} - Outliers {model}",
            model=model,
            source_table=source_table,
            # The frame is aliased to x/y by query_series_frame_for_hide.
            x_col="x",
            y_col="y",
            x=np.asarray(x_out, dtype=float),
            y=np.asarray(y_out, dtype=float),
            outlier_x=np.asarray(x_sorted[mask_sorted], dtype=float),
            outlier_y=np.asarray(y_sorted[mask_sorted], dtype=float),
            outlier_rowids=np.asarray(outlier_rowids, dtype=int),
            metadata={
                "figure_id": self._figure_id,
                "source_series_id": choice["id"],
                "source_sql_query": choice["sql_query"],
                "model": model,
                "threshold": float(params.get("threshold", 3.0)),
                "iqr_factor": float(params.get("iqr_factor", 1.5)),
                "window": int(params.get("window", 11)),
            },
            outlier_count=outlier_count,
            message=message,
        )

    def _params(self) -> dict[str, Any]:
        return self.parameter_values()

    def _outlier_mask(self, y_data: np.ndarray, model: str, params: Mapping[str, Any]) -> np.ndarray:
        threshold = float(params.get("threshold", 3.0))
        if model == OUTLIER_ZSCORE:
            mean = float(np.mean(y_data))
            std = float(np.std(y_data, ddof=0))
            if std <= 0.0:
                return np.zeros_like(y_data, dtype=bool)
            return np.abs((y_data - mean) / std) > threshold
        if model == OUTLIER_IQR:
            q1 = float(np.percentile(y_data, 25.0))
            q3 = float(np.percentile(y_data, 75.0))
            iqr = q3 - q1
            if iqr <= 0.0:
                return np.zeros_like(y_data, dtype=bool)
            factor = float(params.get("iqr_factor", 1.5))
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            return (y_data < lower) | (y_data > upper)
        if model == OUTLIER_MAD:
            median = float(np.median(y_data))
            mad = float(np.median(np.abs(y_data - median)))
            if mad <= 0.0:
                std = float(np.std(y_data, ddof=0))
                if std <= 0.0:
                    return np.zeros_like(y_data, dtype=bool)
                return np.abs(y_data - median) > threshold * std
            scaled = 1.4826 * mad
            return np.abs(y_data - median) > threshold * scaled
        if model == OUTLIER_ROLLING:
            window = self._odd_window(int(params.get("window", 11)), y_data.size)
            median_values = median_filter(y_data, size=window, mode="reflect")
            residuals = np.abs(y_data - median_values)
            scale = float(np.median(residuals))
            if scale <= 0.0:
                std = float(np.std(residuals, ddof=0))
                if std <= 0.0:
                    return np.zeros_like(y_data, dtype=bool)
                scale = std
            return residuals > threshold * scale
        applogger.error(f"Unsupported outlier detection model: {model}")
        return np.zeros(0)

    @staticmethod
    def _odd_window(value: int, n_values: int, minimum: int = 3) -> int:
        window = max(minimum, int(value))
        if window % 2 == 0:
            window += 1
        if window > n_values:
            window = n_values if n_values % 2 == 1 else n_values - 1
        return max(minimum, window)


    @staticmethod
    def _format_results(results: Sequence[OutlierResult]) -> str:
        lines: list[str] = []
        for result in results:
            lines.append(result.source_name)
            lines.append(result.message)
            lines.append(f"Source table: {result.source_table}")
            lines.append(f"Outliers: {result.outlier_count}")
            lines.append(f"Rows marked Hide=True on apply: {result.outlier_count}")
            lines.append("")
        return "\n".join(lines).strip()

    def result_to_frame(self, result: OutlierResult) -> pd.DataFrame:
        return result.to_frame()

    def result_series_spec(self, axis_id: int, table_name: str, result: OutlierResult) -> ResultSeriesSpec:
        del axis_id, table_name, result
        raise NotImplementedError("Outlier dialog marks source-table Hide flags and does not create generated series.")

    @property
    def generated_style_filter(self) -> Mapping[str, Any]:
        return {"generated_outlier_removal": True, "outlier_dialog": "series_outliers"}

    def result_table_name(self, axis_id: int, result: OutlierResult) -> str:
        del axis_id, result
        return generated_table_name("Outlier_Hide_Flags")

    def format_results(self, results: Sequence[OutlierResult]) -> str:
        return self._format_results(results)

    @staticmethod
    def _quote_ident_local(name: str) -> str:
        """Kept as a local name; the implementation is the shared one."""
        return quote_identifier(name)

    def _ensure_preview_state_attrs(self) -> None:
        """Create preview bookkeeping attributes if an older instance lacks them."""
        if not hasattr(self, "_preview_hide_snapshots"):
            self._preview_hide_snapshots: dict[str, list[int]] = {}
        if not hasattr(self, "_preview_series_sql"):
            self._preview_series_sql: dict[int, str] = {}
        if not hasattr(self, "_preview_state_tables"):
            self._preview_state_tables: set[str] = set()

    def _hidden_rowids(self, table_name: str) -> list[int]:
        self._repo.ensure_hide_column(table_name)
        con = getattr(self._repo, "_con", None)
        if con is None:
            return []
        quoted = self._quote_ident_local(table_name)
        rows = con.execute(f'SELECT rowid FROM {quoted} WHERE "Hide" = 1').fetchall()
        return [int(row[0]) for row in rows]

    def _snapshot_outlier_state(self, results: Sequence[OutlierResult]) -> None:
        self._ensure_preview_state_attrs()
        for result in results:
            if result.source_table not in self._preview_hide_snapshots:
                self._preview_hide_snapshots[result.source_table] = self._hidden_rowids(result.source_table)
            source_series_id = result.metadata.get("source_series_id")
            if source_series_id is not None:
                series_id = int(source_series_id)
                if series_id not in self._preview_series_sql:
                    self._preview_series_sql[series_id] = str(result.metadata.get("source_sql_query", ""))

    def _apply_hide_results(self, results: Sequence[OutlierResult]) -> None:
        cleared_tables: set[str] = set()
        for result in results:
            clear_existing = result.source_table not in cleared_tables
            self._repo.mark_hide_rowids(
                table_name=result.source_table,
                rowids=[int(rowid) for rowid in result.outlier_rowids],
                clear_existing=clear_existing,
            )
            cleared_tables.add(result.source_table)

            source_series_id = result.metadata.get("source_series_id")
            if source_series_id is not None:
                self._repo.update_series_hide_filter(
                    int(source_series_id),
                    str(result.metadata.get("source_sql_query", "")),
                )

    def _prepare_preview_state_columns(self, results: Sequence[OutlierResult]) -> None:
        """Snapshot Hide/ClusterId to _Hide/_ClusterId before mutating source tables."""
        self._ensure_preview_state_attrs()
        for result in results:
            table_name = str(result.source_table)
            if table_name not in self._preview_state_tables:
                self._repo.ensure_preview_state_columns(table_name)
                self._preview_state_tables.add(table_name)

    def preview(self) -> bool:
        """Temporarily apply Hide flags so the chart updates, without closing."""
        self._ensure_preview_state_attrs()
        try:
            # If Preview is clicked repeatedly, first restore the source table to
            # the pre-preview snapshot, but keep _Hide/_ClusterId as the original baseline.
            if self._preview_active:
                for table_name in list(self._preview_state_tables):
                    self._repo.restore_preview_state_columns(table_name)
                for series_id, sql_query in self._preview_series_sql.items():
                    self._repo.update_series_sql_query(int(series_id), str(sql_query))

            results = list(self.compute_results())
            if not results:
                show_message(self, "series.no_series_selected", title=self.operation_label)
                return False
            self._snapshot_outlier_state(results)
            self._prepare_preview_state_columns(results)
            self._apply_hide_results(results)
            self.store_cached_results(results)
            self._preview_active = True
            self.applied.emit()
            message = f"Preview updated: Hide column updated for {len(results)} series."
            detail = self.format_results(results)
            self.set_results_text(f"{detail}\n\n{message}" if detail else message)
            return True
        except Exception as exc:
            applogger.exception("Failed to preview outlier hide flags")
            show_message(self, "series.preview_failed", error=exc)
            return False

    def ok(self) -> None:
        """Accept previewed Hide flags, then remove _Hide/_ClusterId."""
        self._ensure_preview_state_attrs()
        if self._preview_active or self.preview():
            for table_name in list(self._preview_state_tables):
                self._repo.drop_preview_state_columns(table_name)
            self._preview_active = False
            self._preview_hide_snapshots.clear()
            self._preview_series_sql.clear()
            self._preview_state_tables.clear()
            self.accept()

    def cancel_operation_changes(self, *, refresh: bool = True) -> None:
        """Restore Hide/ClusterId from _Hide/_ClusterId and restore source SQL."""
        self._ensure_preview_state_attrs()
        if not self._preview_active and not self._preview_state_tables:
            return
        for table_name in list(self._preview_state_tables):
            self._repo.restore_preview_state_columns(table_name)
            self._repo.drop_preview_state_columns(table_name)
        for series_id, sql_query in self._preview_series_sql.items():
            self._repo.update_series_sql_query(int(series_id), str(sql_query))
        self._preview_active = False
        self._preview_hide_snapshots.clear()
        self._preview_series_sql.clear()
        self._preview_state_tables.clear()
        if refresh:
            self.applied.emit()

    def apply(self) -> bool:
        """Compatibility: Preview is the non-closing apply operation for outliers."""
        return self.preview()
