"""Minimal chart-rendering debug window.  Run it directly:

    python _testChart.py

Deliberately the smallest thing that puts a Matplotlib figure in a panel and
keeps it fitted to that panel:

* one QFrame, one FigureCanvasQTAgg, a layout with no margins;
* on resize, the figure is told its new size in inches and redrawn;
* nothing else.

No scroll area, no event filter, no stylesheet, no zoom, no toolbar - and no
imports from ``app``, so it runs even when the application does not.  That is
the point: ChartPanel does all of those things, and when it renders badly the
question is which of them is responsible.

How to use it
-------------
If this window is clean and ChartPanel is not, the fault is in what ChartPanel
adds, and the list of suspects is short.  If this window is *also* wrong, the
problem is below the application - the backend, the Qt build, or the display.

**Translucent** is the one switch worth having here.  ``FigureCanvasQTAgg``
sets ``WA_OpaquePaintEvent`` - "I paint every pixel of my rect, do not erase it
first".  Setting ``WA_TranslucentBackground`` as well is the opposite promise,
and with both set nothing ever clears the canvas rect while Agg blits an RGBA
buffer into it.  Windows returns a clean surface by luck; macOS keeps the
previous frame, and redraws pile up as ghost axes and thickening text.  Press
**Redraw x20** with it on, then off.

**Save PNG** grabs the widget's backing store, not the figure.  ``savefig``
would go back through Agg and produce a clean image on a machine whose screen
is showing residue, which is exactly the trap this is here to avoid.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class FitCanvasPanel(QFrame):
    """A panel whose figure is always exactly the size of the panel."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._canvas = FigureCanvasQTAgg(Figure())

        layout = QVBoxLayout(self)
        # No margins: any gap here is panel background showing around the
        # canvas, which is a second surface someone has to paint.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._draws = 0
        self.plot()

    @property
    def canvas(self) -> FigureCanvasQTAgg:
        return self._canvas

    def set_translucent(self, translucent: bool) -> None:
        """Toggle the attribute that stops the canvas rect being cleared."""
        self._canvas.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground, translucent
        )
        self._canvas.setStyleSheet("background: transparent;" if translucent else "")
        self._canvas.update()

    def plot(self) -> None:
        """Clear and redraw with different data each time.

        ``clear`` matters: adding axes instead of replacing them stacks them,
        which looks identical to a stale buffer and is easier to do by accident.
        """
        self._draws += 1
        x = np.linspace(0.0, 100.0, 12)
        y = x * 0.98 + np.sin(self._draws) * 3.0

        figure = self._canvas.figure
        figure.clear()
        axes = figure.add_subplot(111)
        axes.errorbar(x, y, yerr=np.linspace(0.4, 2.4, x.size), marker="o")
        axes.set_xlabel("applied (units)")
        axes.set_title(f"draw #{self._draws}")
        figure.tight_layout()
        self._canvas.draw_idle()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Fit the figure to the panel.

        The device pixel ratio is the whole subtlety here, and getting it wrong
        is what produced the artifacts this window was written to investigate.

        ``self.width()`` is in *logical* pixels.  ``figure.dpi`` is not the dpi
        you set: the Qt backend multiplies it by the display's pixel ratio, so
        on a 2x screen a 100 dpi figure reports 200.  Dividing logical pixels by
        that dpi therefore gives half the inches needed, the Agg buffer ends up
        a quarter of the widget's area, and the rest of the widget is never
        blitted - it keeps whatever was on screen before.

        Matplotlib's own ``FigureCanvasQT.resizeEvent`` does this correctly:

            w = event.size().width() * self.device_pixel_ratio
            winch = w / self.figure.dpi

        which is worth knowing for a second reason - it means the canvas
        already fits its figure to its widget, and code like this method is
        usually not needed at all.
        """
        super().resizeEvent(event)

        figure = self._canvas.figure
        ratio = self._canvas.device_pixel_ratio or 1.0
        dpi = figure.get_dpi()

        figure.set_size_inches(
            max(self.width(), 1) * ratio / dpi,
            max(self.height(), 1) * ratio / dpi,
            forward=False,
        )
        figure.tight_layout()
        self._canvas.draw_idle()


class DebugDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Chart rendering debug")
        self.resize(760, 520)

        self._panel = FitCanvasPanel(self)
        self._status = QLabel(self)

        self._translucent = QCheckBox("Translucent canvas (reproduces the bug)", self)
        self._translucent.toggled.connect(self._panel.set_translucent)

        root = QVBoxLayout(self)
        root.addWidget(self._panel, 1)
        root.addWidget(self._status)
        root.addWidget(self._translucent)

        buttons = QHBoxLayout()
        for label, slot in (
            ("Redraw", self._redraw),
            ("Redraw x20", self._redraw_many),
            ("Resize", self._toggle_size),
            ("Save PNG", self._save_png),
            ("Close", self.accept),
        ):
            button = QPushButton(label, self)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        root.addLayout(buttons)

        self._report()

    def _redraw(self) -> None:
        self._panel.plot()
        self._report()

    def _redraw_many(self) -> None:
        """Repeated redraws are how a stale buffer becomes visible."""
        for _ in range(20):
            self._panel.plot()
            QApplication.processEvents()
        self._report()

    def _toggle_size(self) -> None:
        """A geometry change is the other way stale pixels show up."""
        wide = self.width() > 700
        self.resize(560 if wide else 760, 400 if wide else 520)
        self._report()

    def _save_png(self) -> None:
        path = Path(tempfile.gettempdir()) / "testchart_backing_store.png"
        self._panel.canvas.grab().save(str(path))
        self._status.setText(str(path))
        print(path)

    def _report(self) -> None:
        canvas = self._panel.canvas
        figure = canvas.figure
        width, height = figure.get_size_inches()
        dpi = figure.get_dpi()
        ratio = canvas.device_pixel_ratio or 1.0
        # Figure size in *logical* pixels, so it can be compared with the panel
        # directly.  If these two disagree, the difference is the strip that
        # never gets painted.
        logical_w = width * dpi / ratio
        logical_h = height * dpi / ratio
        self._status.setText(
            f"panel {self._panel.width()}x{self._panel.height()} | "
            f"figure {logical_w:.0f}x{logical_h:.0f} logical "
            f"({width * dpi:.0f}x{height * dpi:.0f} device) | "
            f"{dpi:.0f} dpi | dpr {ratio:.1f} | axes {len(figure.axes)}"
        )


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = DebugDialog()
    dialog.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
