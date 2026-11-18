"""The list of plots in the sidebar: one editable row per plot.

It listens to scene events, so it stays in sync no matter whether a plot was
added from the panel, the command bar, or anywhere else.
"""

from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from ..app import App
from ..scene import Plot


class PlotRow(QWidget):
    def __init__(self, app: App, plot: Plot, parent=None):
        super().__init__(parent)
        self.app = app
        self.plot = plot

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)

        self.swatch = QLabel()
        self.swatch.setFixedSize(10, 10)

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
        row.addWidget(self.edit, 1)
        row.addWidget(remove)
        self.show_plot(plot)

    def show_plot(self, plot: Plot) -> None:
        self.plot = plot
        self.edit.setText(plot.text)
        r, g, b = (int(c * 255) for c in plot.color)
        self.swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border-radius: 5px;")

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
