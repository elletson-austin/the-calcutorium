"""The list of plots in the sidebar: one editable row per plot.

It listens to scene events, so it stays in sync no matter whether a plot was
added from the panel, the command bar, or anywhere else.
"""

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..app import App
from ..scene import Contour, Plot


class PlotRow(QWidget):
    def __init__(self, app: App, plot: Plot, parent=None):
        super().__init__(parent)
        self.app = app
        self.plot = plot

        rows = QVBoxLayout(self)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(5)
        rows.addLayout(row)

        self.swatch = QLabel()
        self.swatch.setFixedSize(10, 10)

        self.kind = QLabel()
        self.kind.setFixedWidth(52)
        self.kind.setStyleSheet("color: #888888; font-size: 10px;")

        self.edit = QLineEdit(plot.text)
        self.edit.returnPressed.connect(self._commit)
        self.edit.editingFinished.connect(self._commit)

        remove = QPushButton("×")
        remove.setFixedWidth(24)
        remove.setToolTip("Remove")
        remove.setStyleSheet(
            "QPushButton { color: #FF6666; border: none; font-weight: bold; background: transparent; }"
            "QPushButton:hover { color: #FF0000; }"
        )
        remove.clicked.connect(lambda: self.app.remove_plot(self.plot))

        row.addWidget(self.swatch)
        row.addWidget(self.kind)
        row.addWidget(self.edit, 1)
        row.addWidget(remove)

        # Contour settings, on a second line (only shown for contours)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0, 1e6)       # 0 is shown as "auto"
        self.interval.setDecimals(3)
        self.interval.setSingleStep(1)
        self.interval.setSpecialValueText("auto")
        self.interval.setKeyboardTracking(False)  # apply on Enter / arrows, not every keystroke
        self.interval.setToolTip("Draw a level curve at every multiple of this (0 = auto)")
        self.interval.valueChanged.connect(self._set_interval)
        self.contour_settings = QWidget()
        settings = QHBoxLayout(self.contour_settings)
        settings.setContentsMargins(67, 0, 29, 0)  # line up under the equation box
        every = QLabel("every")
        settings.addWidget(every)
        settings.addWidget(self.interval, 1)
        rows.addWidget(self.contour_settings)

        self.show_plot(plot)

    def show_plot(self, plot: Plot) -> None:
        self.plot = plot
        self.kind.setText(plot.kind)  # can change: editing a curve into x*y makes a surface
        self.edit.setText(plot.text)
        r, g, b = (int(c * 255) for c in plot.color)
        self.swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border-radius: 5px;")

        self.contour_settings.setVisible(isinstance(plot, Contour))
        if isinstance(plot, Contour):
            self.interval.blockSignals(True)  # showing the value isn't the user changing it
            self.interval.setValue(plot.interval or 0)
            self.interval.blockSignals(False)

    def _set_interval(self, value: float) -> None:
        if self.app.set_contour_interval(self.plot, value or None) is None:
            self.show_plot(self.plot)  # rejected: show the current value again

    def _commit(self) -> None:
        text = self.edit.text().strip()
        if text == self.plot.text:
            return
        if self.app.edit_plot(self.plot, text) is None:
            self.edit.setText(self.plot.text)  # invalid: put the old equation back


class PlotList(QWidget):
    def __init__(self, app: App, parent=None):
        super().__init__(parent)
        self.app = app
        self.rows: dict[Plot, PlotRow] = {}
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(4)
        self.layout_.addStretch(1)

        app.scene.added.subscribe(self._on_added)
        app.scene.removed.subscribe(self._on_removed)
        app.scene.replaced.subscribe(self._on_replaced)

    def _on_added(self, obj) -> None:
        if isinstance(obj, Plot):
            row = self.rows[obj] = PlotRow(self.app, obj)
            self.layout_.insertWidget(self.layout_.count() - 1, row)

    def _on_removed(self, obj) -> None:
        row = self.rows.pop(obj, None)
        if row is not None:
            row.deleteLater()

    def _on_replaced(self, old, new) -> None:
        row = self.rows.pop(old, None)
        if row is not None:
            self.rows[new] = row
            row.show_plot(new)
