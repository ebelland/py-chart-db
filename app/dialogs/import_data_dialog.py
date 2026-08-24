"""Dialog for importing CSV/Excel files into database tables.

Handles source selection, read options (delimiter, encoding, header, skipped
rows), per-column type overrides, and a live preview of the resulting table.
The import itself is delegated to ``app.utils.import_runner`` so that a saved
link can later be refreshed through exactly the same code path.
"""
from __future__ import annotations

import csv
import re

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from collections.abc import Callable
from typing import Optional

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from app.utils.config import get_import_data_dialog_config, set_import_data_dialog_config
from app.data.sqlite_repo import SqliteRepo
from app.widgets.table_preview import TablePreviewPanel
from app.logs.logger import applogger
from app.utils.messages import show_message
from app.styles.style import (
    apply_dialog_shell,
    create_card_widget,
    create_action_button,
    load_icon,
    mark_editor_panel,
    stdSizeAndlayout,
)
from app.utils.i18n import _


# -----------------------------------------------------------------------------
# Reading helpers
# -----------------------------------------------------------------------------


#: The default table name for pasted data, and the stand-in for a file path
#: wherever one is expected.  Named once: it is both what the table is called
#: and what _safe_table_name_from_filename has to recognise as "not a path".
CLIPBOARD_SOURCE_NAME: str = "from_clipboard"


def sniff_delimiter(sample: str) -> str:
    """Best-effort delimiter detection."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
        return dialect.delimiter
    except Exception:  # noqa: BLE001
        candidates = [",", "\t", ";", "|", " "]
        counts: dict[str, int] = {d: sample.count(d) for d in candidates}
        return max(candidates, key=lambda d: counts[d])


def read_text_file(
    path: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: bool = True,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    encodings = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    encodings = [e for e in encodings if e]

    with open(path, "rb") as f:
        raw = f.read(64 * 1024)

    sample = ""
    for enc in encodings:
        try:
            sample = raw.decode(enc)
            break
        except Exception:  # noqa: BLE001
            continue

    delim = delimiter or sniff_delimiter(sample)
    hdr = 0 if header else None

    last_exc: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(
                path,
                sep=delim,
                encoding=enc,
                skiprows=int(skiprows),
                skipfooter=int(skipfooter),
                engine="python" if skipfooter else "c",
                header=hdr,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    applogger.error(f"Failed reading text file: {last_exc}")
    return pd.DataFrame()


def read_clipboard_text(
    clipboard_text: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: bool = True,
    delimiter: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    text = (clipboard_text or "").strip("\ufeff\n\r\t ")
    if not text:
        return None

    sample = text[:4096]
    delim = delimiter or sniff_delimiter(sample)
    hdr = 0 if header else None

    return pd.read_csv(
        StringIO(text),
        sep=delim,
        skiprows=int(skiprows),
        skipfooter=int(skipfooter),
        engine="python" if skipfooter else "c",
        header=hdr,
    )


def read_excel_file(
    path: str,
    *,
    skiprows: int = 0,
    skipfooter: int = 0,
    sheet_name: Optional[str] = None,
    header: bool = True,
) -> pd.DataFrame:
    hdr = 0 if header else None
    df = pd.read_excel(path, sheet_name=sheet_name or 0, skiprows=int(skiprows), header=hdr, engine="openpyxl")
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df


def read_json_file(path: str, *, skiprows: int = 0, skipfooter: int = 0) -> pd.DataFrame:
    try:
        df = pd.read_json(path)
    except ValueError:
        df = pd.read_json(path, lines=True)

    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df


def read_xml_file(path: str, *, skiprows: int = 0, skipfooter: int = 0) -> pd.DataFrame:
    df = pd.read_xml(path)
    if skiprows:
        df = df.iloc[int(skiprows) :]
    if skipfooter:
        df = df.iloc[: max(0, len(df) - int(skipfooter))]
    return df


#: What the readers above can open, and therefore what the file dialog offers
#: and what a drop onto the main window is accepted for.  One list: the dialog
#: filter and the drop test used to be two literals, and a format added to the
#: reader was a format the window still refused.
IMPORTABLE_SUFFIXES: tuple[str, ...] = (
    ".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".json", ".xml",
)

#: The same list as a QFileDialog filter.
IMPORT_FILE_FILTER: str = (
    "Data files ("
    + " ".join(f"*{suffix}" for suffix in IMPORTABLE_SUFFIXES)
    + ");;All files (*.*)"
)


def is_importable(path: str | Path) -> bool:
    """True when this file is one the import dialog can read."""
    return (Path(path).suffix or "").lower() in IMPORTABLE_SUFFIXES


def read_any_file(
    path: str,
    *,
    skiprows: int,
    skipfooter: int,
    header: bool,
    sheet: Optional[str],
    delim: Optional[str],
    encoding: Optional[str],
) -> pd.DataFrame:
    ext = (Path(path).suffix or "").lower()
    if ext in (".csv", ".tsv", ".txt"):
        return read_text_file(
            path,
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=header,
            encoding=encoding,
            delimiter=delim,
        )
    if ext in (".xlsx", ".xlsm", ".xls"):
        return read_excel_file(path, skiprows=skiprows, skipfooter=skipfooter, sheet_name=sheet, header=header)
    if ext in (".json",):
        return read_json_file(path, skiprows=skiprows, skipfooter=skipfooter)
    if ext in (".xml",):
        return read_xml_file(path, skiprows=skiprows, skipfooter=skipfooter)

    # fallback: try delimited text
    return read_text_file(
        path,
        skiprows=skiprows,
        skipfooter=skipfooter,
        header=header,
        encoding=encoding,
        delimiter=delim,
    )


# -----------------------------------------------------------------------------
# Preview model
# -----------------------------------------------------------------------------


class DataFramePreviewModel(QAbstractTableModel):
    """Lightweight preview model for DataFrames."""

    def __init__(self, df: pd.DataFrame, *, max_rows: int = 500, max_cols: int = 200, parent=None) -> None:
        super().__init__(parent)
        self._df = df
        self._cols = list(df.columns)[:max_cols]
        self._nrows = min(len(df), max_rows)

    def rowCount(self, parent: QModelIndex|QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self._nrows

    def columnCount(self, parent: QModelIndex|QPersistentModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._cols)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._cols[section]) if 0 <= section < len(self._cols) else None
        return str(section + 1)

    def data(self, index: QModelIndex|QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        r = index.row()
        c = index.column()
        if r >= self._nrows or c >= len(self._cols):
            return None
        col = self._cols[c]
        val = self._df.iloc[r][col]
        return "" if pd.isna(val) else str(val)


# -----------------------------------------------------------------------------
# Import dialog
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ImportResult:
    table_name: str
    rows: int
    cols: int


class ImportDataDialog(QDialog):
    """Import data from files or clipboard into the current SQLite database."""

    SQLITE_TYPES = [
        "Ignore",
        "INTEGER",
        "REAL",
        "TEXT",
        "BLOB",
        "NUMERIC",
        "DATE",
        "TIME",
        "DATETIME",
    ]

    ENCODINGS = [
        "auto",
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
        "utf-16",
    ]

    def __init__(self, repo: SqliteRepo, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.result: Optional[ImportResult] = None
        self._df: Optional[pd.DataFrame] = None

        # Which of the two sources the preview and the import read from.
        # Every path that sets one must clear the other, or the dialog shows
        # one source's data under the other's name - which is exactly what
        # pasting after opening a file used to do.
        self._source_mode: str = "none"  # none|file|clipboard
        self._clipboard_text: str = ""
        self._path: str = ""
        self._last_auto_table: str = ""

        self.setWindowTitle(_("Import data"))
        self.setWindowIcon(load_icon("import"))
        # Size and root padding come from the shared dialog shell.

        cfg = get_import_data_dialog_config()
        self._desired_sheet = str(cfg.get("sheet", "") or "").strip()

        # ---------------- Left panel (compact) ----------------
        left = create_card_widget(self, "importOptionsCard")
        left_layout = QVBoxLayout(left)
        stdSizeAndlayout(left_layout)

        form = QFormLayout()
        stdSizeAndlayout(form)

        # Source row: path + Browse + Clipboard
        src_row = QWidget(left)
        src_lay = QVBoxLayout(src_row)
        stdSizeAndlayout(src_lay)

        src_top = QWidget(src_row)
        src_top_lay = QHBoxLayout(src_top)
        stdSizeAndlayout(src_top_lay)

        self._btn_browse = create_action_button(
                               parent=src_top,
                               action_id="open",
                               action=self._on_browse,
                               layout=src_top_lay,
                           )
        self._btn_clip = create_action_button(
                             parent=src_top,
                             action_id="paste",
                             action=self._on_load_clipboard,
                             layout=src_top_lay,
                         )
        src_lay.addWidget(src_top)

        form.addRow(_("Source"), src_row)

        # Excel worksheet selector (shown only for Excel files)
        self._sheet = QComboBox(left)
        self._sheet.setEnabled(False)
        self._sheet.setVisible(False)
        self._sheet.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._sheet.setToolTip(_("Worksheet (for Excel sources)"))
        stdSizeAndlayout(self._sheet)
        form.addRow(_("Sheet"), self._sheet)
        self._sheet_label = form.labelForField(self._sheet)
        if self._sheet_label is not None:
            self._sheet_label.setVisible(False)

        # Table name
        self._table = QLineEdit(left)
        self._table.setText(str(cfg.get("table", "")))
        stdSizeAndlayout(self._table)
        form.addRow(_("Table"), self._table)

        # Header
        self._has_header = QCheckBox(_("First row is header"), left)
        self._has_header.setChecked(bool(cfg.get("header", True)))
        stdSizeAndlayout(self._has_header)
        form.addRow("", self._has_header)

        # Skip rows (top)
        self._skip_rows = QSpinBox(left)
        self._skip_rows.setRange(0, 1_000_000)
        self._skip_rows.setValue(int(cfg.get("skip_rows", 0) or 0))
        self._skip_rows.setKeyboardTracking(False)
        self._skip_rows.setAccelerated(True)
        self._skip_rows.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        # If global QSS breaks hit-testing, neutralize for this widget
        stdSizeAndlayout(self._skip_rows)
        form.addRow(_("Skip top rows"), self._skip_rows)

        # Skip rows (bottom)
        self._skip_last = QSpinBox(left)
        self._skip_last.setRange(0, 1_000_000)
        self._skip_last.setValue(int(cfg.get("skip_last", 0) or 0))
        self._skip_last.setKeyboardTracking(False)
        self._skip_last.setAccelerated(True)
        self._skip_last.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        stdSizeAndlayout(self._skip_last)
        form.addRow(_("Skip last rows"), self._skip_last)

        # Delimiter dropdown (editable)
        self._delim = QComboBox(left)
        self._delim.setEditable(True)
        self._delim.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._delim.addItem(_("auto"), userData=None)
        self._delim.addItem(",", userData=",")
        self._delim.addItem(";", userData=";")
        # The escape sequence itself, which reads the same in every language.
        self._delim.addItem("\\t", userData="\t")
        self._delim.addItem("|", userData="|")
        self._delim.addItem(_("space"), userData=" ")
        saved_delim = (cfg.get("delim", "") or "").strip()
        self._delim.setCurrentIndex(0 if not saved_delim else 0)
        if saved_delim:
            self._delim.setEditText(saved_delim)
        stdSizeAndlayout(self._delim)
        form.addRow(_("Delimiter"), self._delim)

        # Encoding dropdown (editable)
        self._encoding = QComboBox(left)
        self._encoding.setEditable(True)
        self._encoding.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for e in self.ENCODINGS:
            self._encoding.addItem(e)
        saved_enc = (cfg.get("encoding", "auto") or "auto").strip() or "auto"
        self._encoding.setCurrentText(saved_enc)
        stdSizeAndlayout(self._encoding)
        form.addRow(_("Encoding"), self._encoding)

        left_layout.addLayout(form)

        # Column mapping table
        self._col_table = QTableWidget(left)
        self._col_table.setColumnCount(2)
        self._col_table.setHorizontalHeaderLabels(["Column", "Type"])
        self._col_table.verticalHeader().setVisible(False)
        self._col_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._col_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._col_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._col_table.horizontalHeader().setStretchLastSection(True)
        mark_editor_panel(self._col_table)
        left_layout.addWidget(self._col_table, 1)

        # Buttons row.  There is no Preview button: the preview already
        # refreshes itself on Browse, on paste, and a moment after any option
        # changes, so pressing it could only ever repeat what just happened.
        btn_row = QWidget(left)
        btn_lay = QHBoxLayout(btn_row)
        stdSizeAndlayout(btn_lay)

        self._btn_ok = create_action_button(
                           parent=btn_row,
                           action_id="apply",
                           action=self._on_accept,
                           layout=btn_lay,
                       )
        self._btn_cancel = create_action_button(
                               parent=btn_row,
                               action_id="close",
                               action=self.reject,
                               layout=btn_lay,
                           )
        self._btn_ok.setDefault(True)
        btn_lay.addStretch(1)

        left_layout.addWidget(btn_row, 0)

        # Make left scrollable for smaller screens
        left_scroll = QScrollArea(self)
        stdSizeAndlayout(left_scroll)
        left_scroll.setWidget(left)

        # ---------------- Right: preview ----------------
        self._preview = TablePreviewPanel(self,self._repo)
        self._preview.refresh.connect(self._schedule_preview)
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self._preview)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 620])

        root = QVBoxLayout(self)
        apply_dialog_shell(self, root, size="medium")
        root.addWidget(splitter, 1)
        self.setLayout(root)

        # ---------------- Events & shortcuts ----------------

        if hasattr(self, "_sheet"):
            self._sheet.currentIndexChanged.connect(lambda _=0: self._on_sheet_changed())  # type: ignore

        # Auto preview ONLY when browse is selected
        self._has_header.stateChanged.connect(self._schedule_preview)  # type: ignore
        self._skip_rows.valueChanged.connect(self._schedule_preview)  # type: ignore
        self._skip_last.valueChanged.connect(self._schedule_preview)  # type: ignore
        self._delim.currentIndexChanged.connect(self._schedule_preview)  # type: ignore
        self._delim.editTextChanged.connect(self._schedule_preview)  # type: ignore
        self._encoding.currentIndexChanged.connect(self._schedule_preview)  # type: ignore
        self._encoding.editTextChanged.connect(self._schedule_preview)  # type: ignore

        # Clipboard shortcut
        self._sc_paste = QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self)
        self._sc_paste.activated.connect(self._on_load_clipboard)  # type: ignore

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._refresh_preview)  # type: ignore

        self.file_name:str=""

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _schedule_preview(self) -> None:
        """Re-read the current source after the options settle.

        Pasted text is re-read too. Header, delimiter and the two skip counts
        describe how to *parse* a source, not where it came from, so a paste
        that landed with the wrong delimiter used to be unfixable without
        pasting it again.
        """
        if self._source_mode == "none":
            return
        self._preview_timer.start(250)

    def _current_delim(self) -> Optional[str]:
        data = self._delim.currentData()
        if data is not None:
            return str(data)
        txt = (self._delim.currentText() or "").strip()
        if not txt or txt.lower() == "auto":
            return None
        if txt.lower() in ("tab", "\\t"):
            return "\t"
        if txt.lower() == "space":
            return " "
        return txt

    def _current_encoding(self) -> Optional[str]:
        txt = (self._encoding.currentText() or "").strip()
        if not txt or txt.lower() == "auto":
            return None
        return txt

    def _is_excel_path(self, path: str) -> bool:
        ext = (Path(path).suffix or "").lower()
        return ext in (".xlsx", ".xlsm", ".xls")

    def _current_sheet(self) -> str | None:
        if not hasattr(self, "_sheet"):
            return None
        if not self._sheet.isVisible() or self._sheet.count() == 0:
            return None
        data = self._sheet.currentData()
        return str(data) if data else (str(self._sheet.currentText() or "") or None)

    def _set_sheet_row_visible(self, visible: bool) -> None:
        self._sheet.setVisible(visible)
        self._sheet.setEnabled(visible)
        self._sheet_label.setVisible(visible)

    def _update_sheet_choices(self, path: str) -> None:
        if not hasattr(self, "_sheet"):
            return

        if not self._is_excel_path(path):
            self._sheet.blockSignals(True)
            try:
                self._sheet.clear()
                self._set_sheet_row_visible(False)
            finally:
                self._sheet.blockSignals(False)
            return

        sheets: list[str] = []
        try:
            ext = (Path(path).suffix or "").lower()
            engine = "openpyxl" if ext in (".xlsx", ".xlsm") else None
            xls = pd.ExcelFile(path, engine=engine)
            sheets = [str(s) for s in (xls.sheet_names or [])]
        except Exception as exc:  # noqa: BLE001
            applogger.warning("Cannot read Excel sheet names for %s: %s", path, exc)
            sheets = []

        self._sheet.blockSignals(True)
        try:
            self._sheet.clear()
            if sheets:
                for s in sheets:
                    self._sheet.addItem(s, userData=s)
                if self._desired_sheet:
                    idx = self._sheet.findData(self._desired_sheet)
                    if idx >= 0:
                        self._sheet.setCurrentIndex(idx)
                self._set_sheet_row_visible(True)
            else:
                self._set_sheet_row_visible(False)
        finally:
            self._sheet.blockSignals(False)

    def _on_sheet_changed(self) -> None:
        if self._source_mode == "file":
            self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Re-read whichever source is current and show it.

        Dispatching on the source mode is the point. This used to read
        ``self.file_name`` and nothing else, so a paste that arrived after a
        file had been opened was overwritten by that file the moment anything
        called this - which the paste handler itself did, on its last line.
        """
        if self._source_mode == "clipboard":
            failure = self._show_frame(self._read_clipboard_source)
            if failure is not None:
                show_message(self, "import.clipboard_failed", error=failure)
            return

        path = (self.file_name or "").strip()
        if not path or not Path(path).exists():
            return

        failure = self._show_frame(lambda: self._read_source(path))
        if failure is not None:
            show_message(self, "import.preview_failed", error=failure)

    def _show_frame(self, read: Callable[[], pd.DataFrame]) -> Exception | None:
        """Read one source through *read*, then fill the columns and preview.

        Shared by the file and the clipboard because the only difference
        between them is the reading; everything after it - cleaning, the type
        table, hiding the empty columns from the preview - is the same work,
        and was the same work written twice.

        The failure is returned rather than reported, so that each caller
        names its own catalogue message at its own call site. The ids have to
        be literals where show_message is called: that is what the sweep
        proving every id is defined, and every defined id used, can see.
        """
        try:
            self._df = read()
            self._apply_skip_last_and_clean()

            # Columns: keep empty columns (default Ignore)
            self._build_columns_table(include_empty=True)

            # Preview: hide empty columns
            df_prev = self._df
            if df_prev is not None:
                empty_cols = self._empty_columns(df_prev)
                if empty_cols:
                    df_prev = df_prev.drop(columns=empty_cols)

            if df_prev is None:
                self._preview.clear()
            else:
                self._preview.set_model(DataFramePreviewModel(df_prev, parent=self._preview.view))

            return None

        except Exception as exc:  # noqa: BLE001
            applogger.exception("Preview failed: %s", exc)
            self._df = None
            self._preview.clear()
            self._col_table.setRowCount(0)
            return exc

    def _read_clipboard_source(self) -> pd.DataFrame:
        """Parse the remembered clipboard text with the current options."""
        return read_clipboard_text(
            self._clipboard_text,
            skiprows=int(self._skip_rows.value()),
            skipfooter=int(self._skip_last.value()),
            header=bool(self._has_header.isChecked()),
            delimiter=self._current_delim(),
        )

    def _read_source(self, path: str) -> pd.DataFrame:
        delim = self._current_delim()
        encoding = self._current_encoding()
        skiprows = int(self._skip_rows.value())
        skipfooter = int(self._skip_last.value())
        header = bool(self._has_header.isChecked())

        return read_any_file(
            path,
            skiprows=skiprows,
            skipfooter=skipfooter,
            header=header,
            sheet=self._current_sheet(),
            delim=delim,
            encoding=encoding,
        )

    def _apply_skip_last_and_clean(self) -> None:
        if self._df is None:
            return
        self._df.columns = [str(c).strip() for c in self._df.columns]
        skip_last = int(self._skip_last.value())
        if skip_last > 0 and len(self._df) > 0:
            self._df = self._df.iloc[: max(0, len(self._df) - skip_last)]

    @staticmethod
    def _empty_columns(df: pd.DataFrame) -> list[str]:
        empties: list[str] = []
        for col in df.columns:
            s = df[col]
            if s.isna().all():
                empties.append(str(col))
                continue
            try:
                ss = s.astype(str).str.strip().replace("nan", "")
                if ss.eq("").all():
                    empties.append(str(col))
            except Exception:
                pass
        return empties

    # ------------------------------------------------------------------
    # Column typing
    # ------------------------------------------------------------------
    def _build_columns_table(self, *, include_empty: bool) -> None:
        df = self._df
        self._col_table.setRowCount(0)
        if df is None or df.empty:
            return

        empty_cols = set(self._empty_columns(df))
        cols = [str(c) for c in df.columns]
        if not include_empty:
            cols = [c for c in cols if c not in empty_cols]

        self._col_table.setRowCount(len(cols))
        for r, col in enumerate(cols):
            it_name = QTableWidgetItem(col)
            it_name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._col_table.setItem(r, 0, it_name)

            combo = QComboBox(self._col_table)
            combo.addItems(self.SQLITE_TYPES)
            if col in empty_cols:
                combo.setCurrentText("Ignore")
            else:
                combo.setCurrentText(self._guess_sqlite_type(df[col]))
            self._col_table.setCellWidget(r, 1, combo)

        self._col_table.resizeColumnsToContents()

    @staticmethod
    def _guess_sqlite_type(series: pd.Series) -> str:
        try:
            s = series.dropna()
            if s.empty:
                return "Ignore"
            if pd.api.types.is_integer_dtype(s):
                return "INTEGER"
            if pd.api.types.is_float_dtype(s):
                return "REAL"
            if pd.api.types.is_bool_dtype(s):
                return "INTEGER"
            if pd.api.types.is_datetime64_any_dtype(s):
                return "DATETIME"
        except Exception:
            pass
        return "TEXT"

    def _selected_types(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for r in range(self._col_table.rowCount()):
            name_it = self._col_table.item(r, 0)
            if not name_it:
                continue
            col = name_it.text()
            combo = self._col_table.cellWidget(r, 1)
            if not isinstance(combo, QComboBox):
                continue
            t = combo.currentText().strip().upper()
            if t == "IGNORE":
                continue
            mapping[col] = t
        return mapping

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _set_file_name_label(self, path: str) -> None:
        p = Path((path or "").strip())
        if not path or not p.name:
            self.file_name=""
            self.setWindowTitle(_("Import data"))
            return
        self.file_name=path
        self.setWindowTitle(f"Import data: {path}")


    def _safe_table_name_from_filename(self,path: str) -> str:
        base = path if path == CLIPBOARD_SOURCE_NAME else Path(path).stem.strip()
        if not base:
            base = "new_table"
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", base)
        if not re.match(r"^[A-Za-z_]", name):
            name = "t_" + name
        name = name.strip("_") or "new_table"
        nname=name
        iter:int=1
        while self._repo.check_if_table_exists(nname):
            nname=name+"_"+str(iter)
            iter+=1    
        return nname

    def _set_default_table_name(self, path: str) -> None:
        auto = self._safe_table_name_from_filename(path)
        self._table.setText(auto)
        self._last_auto_table = auto

    def _on_browse(self) -> None:
        path, _unused = QFileDialog.getOpenFileName(
            self,
            _("Select file"),
            str(Path.home()),
            IMPORT_FILE_FILTER,
        )
        if not path:
            return
        self.load_file(path)

    def load_file(self, path: str | Path) -> None:
        """Show *path* as the source, exactly as Browse does.

        Public because Browse is no longer the only way a file arrives: the
        main window accepts a drop and opens this dialog on what was dropped.
        Sharing the method rather than the four lines means the dropped file
        gets the sheet list, the default table name and the preview too - the
        parts that are easy to leave out of a second copy.
        """
        name = str(path)
        self._source_mode = "file"
        self._set_file_name_label(name)
        self._set_default_table_name(name)
        self._path = name
        self._update_sheet_choices(name)

        # Auto preview immediately: the file was chosen, not typed.
        self._refresh_preview()

    def _on_load_clipboard(self) -> None:
        """Replace whatever is loaded with what is on the clipboard.

        *Replace* is the whole job. The clipboard becomes the source, so the
        file - if one was opened - stops being it: the remembered path, the
        window title and the sheet list all go, and the table name is taken
        from the clipboard rather than left naming a file the preview no
        longer shows.

        The text is kept rather than only its parse. Header, delimiter and
        the skip counts can be changed afterwards, and re-reading the source
        is how they take effect - which needs the source still to be here.
        """
        text = QApplication.clipboard().text() or ""

        # Parsed before anything is replaced: an empty clipboard must leave
        # the dialog exactly as it was, not clear it and then say why.
        try:
            probe = read_clipboard_text(
                text,
                skiprows=int(self._skip_rows.value()),
                skipfooter=int(self._skip_last.value()),
                header=bool(self._has_header.isChecked()),
                delimiter=self._current_delim(),
            )
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Clipboard preview failed: %s", exc)
            show_message(self, "import.clipboard_failed", error=exc)
            return

        if probe is None or probe.empty:
            show_message(self, "import.clipboard_empty")
            return

        self._source_mode = "clipboard"
        self._clipboard_text = text
        self._path = ""
        self._set_file_name_label("")
        self._update_sheet_choices("")
        self._set_default_table_name(CLIPBOARD_SOURCE_NAME)

        self._refresh_preview()

    def _on_accept(self) -> None:
        if self._df is None:
            # If user did not press Preview, read once for import.
            self._refresh_preview()

        df = self._df
        if df is None or df.empty:
            applogger.info("Import aborted: no data to import")
            show_message(self, "import.nothing_to_import")
            return

        table = (self._table.text() or "").strip()
        if not table:
            show_message(self, "import.no_table_name")
            return

        types = {k: v for k, v in self._selected_types().items() if v.lower() != "ignore"}
        cols_keep = [c for c in df.columns if c in types]
        if not cols_keep:
            applogger.info("Import aborted: no columns to import")
            show_message(self, "import.no_columns_selected")
            return

        df2 = df[cols_keep].copy()
        df2 = self._coerce_df(df2, types)
        set_import_data_dialog_config(
            {
                "table": table,
                "header": bool(self._has_header.isChecked()),
                "skip_rows": int(self._skip_rows.value()),
                "skip_last": int(self._skip_last.value()),
                "delim": self._current_delim() or "",
                "encoding": self._current_encoding() or "auto",
                "sheet": self._current_sheet() or "",
            }
        )
        try:
            table_name=self._safe_table_name_from_filename(table)
            self._repo.import_into_sqlite(table_name, df2, types)
            if self._source_mode == "file":
                cfg=get_import_data_dialog_config()
                if not self._repo.upsert_link(table_name= table_name, source_path=self._path, settings=cfg):
                    applogger.warning("Failed to create link for imported table '%s'", table_name)
        except Exception as exc:  # noqa: BLE001
            applogger.exception("Import failed: %s", exc)
            show_message(self, "import.failed", error=exc)
            return

        self.result = ImportResult(table_name=table, rows=int(len(df2)), cols=int(len(df2.columns)))
        self.accept()

    @staticmethod
    def _coerce_df(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
        out = df.copy()
        for col, t in types.items():
            if col not in out.columns:
                continue
            tt = t.upper()
            s = out[col]

            if tt == "INTEGER":
                out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            elif tt == "REAL":
                out[col] = pd.to_numeric(s, errors="coerce")
            elif tt in ("DATE", "TIME", "DATETIME"):
                dt = pd.to_datetime(s, errors="coerce")
                if tt == "DATE":
                    out[col] = dt.dt.date.astype("string")
                elif tt == "TIME":
                    out[col] = dt.dt.time.astype("string")
                else:
                    out[col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")
            elif tt == "BLOB":
                out[col] = s.apply(
                    lambda v: v
                    if isinstance(v, (bytes, bytearray))
                    else ("" if pd.isna(v) else str(v)).encode("utf-8")
                )
            else:
                out[col] = s.astype("string")

        return out