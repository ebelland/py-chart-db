"""The import dialog's two sources, and the rule that only one is current.

Opening a file and pasting are alternatives, not layers. The dialog kept the
file's path, title and sheet list when data was pasted over it, and then - on
the last line of the paste handler - called the preview refresh, which read
``self.file_name`` and nothing else. So pasting after opening a file put the
file's rows back under the clipboard's table name, and pasting again changed
nothing at all.

These tests are mostly about which source is current after each move, because
that is the state the bug lived in.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.data.sqlite_repo import SqliteRepo
from app.dialogs.import_data_dialog import CLIPBOARD_SOURCE_NAME, ImportDataDialog


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    path.write_text("region,units\nnorth,10\nsouth,20\n", encoding="utf-8")
    return path


@pytest.fixture
def dialog(qapp, tmp_db_path: Path):
    repo = SqliteRepo(db_path=tmp_db_path)
    built = ImportDataDialog(repo)
    yield built
    repo.close()


def _paste(dialog, text: str) -> None:
    QApplication.clipboard().setText(text)
    dialog._on_load_clipboard()


def _columns(dialog) -> list[str]:
    assert dialog._df is not None
    return [str(column) for column in dialog._df.columns]


# ----------------------------------------------------------------------
# Replacing one source with the other
# ----------------------------------------------------------------------
def test_pasting_over_an_opened_file_replaces_it(dialog, csv_file: Path) -> None:
    """The bug, in one test: the file's rows came back after the paste."""
    dialog.load_file(csv_file)
    assert _columns(dialog) == ["region", "units"]

    _paste(dialog, "alpha\tbeta\n1\t2\n")

    assert dialog._source_mode == "clipboard"
    assert _columns(dialog) == ["alpha", "beta"]


def test_pasting_again_replaces_the_previous_paste(dialog) -> None:
    _paste(dialog, "alpha\tbeta\n1\t2\n")
    _paste(dialog, "x\ty\tz\n7\t8\t9\n")

    assert _columns(dialog) == ["x", "y", "z"]
    assert len(dialog._df) == 1


def test_pasting_forgets_the_file_it_replaced(dialog, csv_file: Path) -> None:
    """Not cosmetic: the remembered path is what the import link is written
    from, and the preview used to read it back."""
    dialog.load_file(csv_file)
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    assert dialog.file_name == ""
    assert dialog._path == ""
    assert dialog._table.text() == CLIPBOARD_SOURCE_NAME


def test_opening_a_file_after_pasting_replaces_the_paste(dialog, csv_file: Path) -> None:
    _paste(dialog, "alpha\tbeta\n1\t2\n")
    dialog.load_file(csv_file)

    assert dialog._source_mode == "file"
    assert _columns(dialog) == ["region", "units"]
    assert dialog._table.text() == "sales"


def test_a_refresh_after_pasting_re_reads_the_clipboard_not_a_file(
    dialog, csv_file: Path
) -> None:
    """_refresh_preview is called from the option timer and the sheet combo
    as well as by hand, so it has to be safe to call at any time."""
    dialog.load_file(csv_file)
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    dialog._refresh_preview()

    assert _columns(dialog) == ["alpha", "beta"]


# ----------------------------------------------------------------------
# The options describe how to parse, not where it came from
# ----------------------------------------------------------------------
def test_the_header_option_re_applies_to_pasted_data(dialog) -> None:
    """Pasting used to be a one-shot parse: getting the options wrong meant
    pasting the whole thing again."""
    _paste(dialog, "alpha\tbeta\n1\t2\n")
    assert _columns(dialog) == ["alpha", "beta"]

    dialog._has_header.setChecked(False)
    dialog._refresh_preview()

    assert _columns(dialog) != ["alpha", "beta"]
    assert len(dialog._df) == 2, "the header row is data now"


def test_skip_rows_re_applies_to_pasted_data(dialog) -> None:
    _paste(dialog, "junk\nalpha\tbeta\n1\t2\n3\t4\n")

    dialog._skip_rows.setValue(1)
    dialog._refresh_preview()

    assert _columns(dialog) == ["alpha", "beta"]
    assert len(dialog._df) == 2


def test_an_option_change_schedules_a_refresh_for_pasted_data_too(dialog) -> None:
    """The timer used to return early unless the source was a file."""
    _paste(dialog, "alpha\tbeta\n1\t2\n")

    dialog._schedule_preview()

    assert dialog._preview_timer.isActive()


def test_nothing_is_scheduled_when_there_is_no_source(dialog) -> None:
    dialog._schedule_preview()

    assert not dialog._preview_timer.isActive()


# ----------------------------------------------------------------------
# Refusing a paste leaves what is loaded alone
# ----------------------------------------------------------------------
def test_an_empty_clipboard_does_not_clear_what_is_loaded(
    dialog, csv_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing the dialog and *then* saying why would be the worst of both."""
    import app.dialogs.import_data_dialog as module

    shown: list[str] = []
    monkeypatch.setattr(
        module, "show_message", lambda _p, message_id, **_k: shown.append(message_id)
    )

    dialog.load_file(csv_file)
    _paste(dialog, "   ")

    assert shown == ["import.clipboard_empty"]
    assert dialog._source_mode == "file"
    assert _columns(dialog) == ["region", "units"]
