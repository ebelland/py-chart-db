"""Manual demo of SeriesOperationDialogBase.

Not a pytest module: run it directly to inspect the shared operation-dialog
shell against a real database.
"""

# ----------------------------------------------------------------------
# Demo / manual test
# ----------------------------------------------------------------------
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.data.sqlite_repo import SqliteRepo
from app.series_operations.dialog_base import SeriesOperationDialogBase
from app.styles.style import apply_platform_style
from main import _select_database


class DemoSeriesOperationDialog(SeriesOperationDialogBase):
    """Minimal implementation used to test the base dialog."""

    def build_model_selector(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Demo model selector"))
        combo = QComboBox(widget)
        combo.addItems(["Model A", "Model B", "Model C"])
        layout.addWidget(combo)

        return widget

    def build_parameter_selector(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Demo parameters"))
        edit = QPlainTextEdit(widget)
        edit.setPlainText("Parameter editor placeholder")
        layout.addWidget(edit)

        return widget

    def refresh_results(self) -> None:
        self.set_results_text(
            "Demo preview\n\n"
            f"Selected series: {len(self.selected_series())}"
        )

    def apply_changes(self) -> None:
        print("Apply clicked")

    def build_result_series(self) -> list:
        return []


def main() -> None:
    import sys
    app = QApplication(sys.argv)
    apply_platform_style(app)
    db_path = _select_database()
    if db_path is None:
        return
    db_path = SqliteRepo.ensure_dhub_extension(db_path)
    repo = SqliteRepo(db_path=db_path)



    dlg = DemoSeriesOperationDialog(
        repo=repo,
        figure_id=1,
        title="Series Operation Base Demo",
    )

    dlg.resize(1200, 800)
    dlg.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
