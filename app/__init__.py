"""ChartLibre.

The name, the version and the mark live here because two dialogs and a
window title need them, and a string repeated in three places is a string
that ends up disagreeing with itself.  Bump ``APP_VERSION`` here and
nowhere else.
"""

APP_NAME: str = "ChartLibre"
APP_VERSION: str = "0.1.0"

#: The application's own mark: an open bracket - "libre", a structure left
#: open to write into rather than a locked box - graduated like a ruler,
#: with a small three-point chart breaking out past it. Bare path data, the
#: same shape ``SeriesOperationDialogBase.Icon`` uses for a plugin's own
#: icon; ``app.styles.style.icon_from_svg_source`` wraps it in the standard
#: stroke-only SVG document at whatever size a window icon needs.
APP_ICON: str = (
    '<path d="M11 5H7v14h4"/>'
    '<path d="M4 9h3M4 15h3"/>'
    '<path d="M10 16l5-6 4-5"/>'
)
