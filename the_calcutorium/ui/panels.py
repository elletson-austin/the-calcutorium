"""Sidebar panels. Each button calls an App action, the same ones typed commands use."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..app import App
from ..planes import PLANES
from .plot_list import PlotList

PANEL_STYLE = """
    QWidget { background-color: #2E2E2E; color: #F0F0F0; }
    QPushButton {
        background-color: #4A4A4A; color: #F0F0F0;
        border: 1px solid #666; padding: 4px 8px;
    }
    QPushButton:hover { background-color: #5A5A5A; }
    QPushButton:pressed { background-color: #3A3A3A; }
    QLineEdit, QDoubleSpinBox, QSpinBox {
        background-color: #3C3C3C; color: #F0F0F0;
        border: 1px solid #555; padding: 3px;
    }
    QGroupBox {
        color: #AAAAAA; border: 1px solid #555555;
        margin-top: 10px; font-weight: bold;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
    QLabel { color: #AAAAAA; }
    QScrollArea { border: none; background-color: #2E2E2E; }
"""


def _spin(value: float, lo: float, hi: float, decimals: int = 2) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    return s


def _int_spin(value: int, lo: int, hi: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


class PlotsPanel(QWidget):
    def __init__(self, app: App, parent=None):
        super().__init__(parent)
        self.app = app
        self.setStyleSheet(PANEL_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        add_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g. z = sin(x)*cos(y)")
        self.input.returnPressed.connect(self._add)
        add_button = QPushButton("Add")
        add_button.setFixedWidth(50)
        add_button.clicked.connect(self._add)
        add_row.addWidget(self.input)
        add_row.addWidget(add_button)
        layout.addLayout(add_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(PlotList(app))
        layout.addWidget(scroll, 1)

    def _add(self) -> None:
        text = self.input.text().strip()
        if text and self.app.add_plot(text) is not None:
            self.input.clear()


class SimulationsPanel(QWidget):
    def __init__(self, app: App, parent=None):
        super().__init__(parent)
        self.app = app
        self.setStyleSheet(PANEL_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._group("lorenz", "Lorenz Attractor", {
            "sigma": _spin(10.0, 0.1, 100.0),
            "rho": _spin(28.0, 0.1, 100.0),
            "beta": _spin(8.0 / 3.0, 0.01, 10.0),
            "dt": _spin(0.001, 0.0001, 0.1, decimals=4),
            "steps": _int_spin(5, 1, 50),
        }))
        layout.addWidget(self._group("nbody", "N-Body Simulation", {
            "num_bodies": _int_spin(4000, 100, 50000),
            "G": _spin(1.0, 0.01, 100.0),
            "dt": _spin(0.01, 0.001, 1.0, decimals=3),
            "softening": _spin(1.0, 0.01, 10.0),
            "steps": _int_spin(5, 1, 50),
        }))
        layout.addStretch(1)

    def _group(self, name: str, title: str, fields: dict) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        buttons = QHBoxLayout()
        add, remove = QPushButton("Add"), QPushButton("Remove")
        add.clicked.connect(lambda: self.app.add_simulation(name))
        remove.clicked.connect(lambda: self.app.remove_simulation(name))
        buttons.addWidget(add)
        buttons.addWidget(remove)
        layout.addLayout(buttons)

        grid = QGridLayout()
        grid.setSpacing(4)
        for row, (label, widget) in enumerate(fields.items()):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)

        apply = QPushButton("Apply")
        apply.clicked.connect(
            lambda: self.app.update_simulation(name, {k: w.value() for k, w in fields.items()})
        )
        layout.addWidget(apply)
        return group


class ViewPanel(QWidget):
    def __init__(self, app: App, parent=None):
        super().__init__(parent)
        self.app = app
        self.setStyleSheet(PANEL_STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._mode_group())
        layout.addWidget(self._range_group())
        layout.addStretch(1)

    def _mode_group(self) -> QGroupBox:
        group = QGroupBox("View")
        layout = QGridLayout(group)
        layout.setSpacing(4)
        button_3d = QPushButton("3D")
        button_3d.clicked.connect(lambda: self.app.set_view("3d"))
        layout.addWidget(button_3d, 0, 0, 1, 3)
        for col, plane in enumerate(PLANES):
            button = QPushButton(plane.upper())
            button.clicked.connect(lambda _=False, p=plane: self.app.set_view(p))
            layout.addWidget(button, 1, col)
        return group

    def _range_group(self) -> QGroupBox:
        group = QGroupBox("Range (2D)")
        layout = QGridLayout(group)
        layout.setSpacing(4)
        for row, axis in enumerate("xyz"):
            lo, hi = _spin(-10.0, -1e4, 1e4), _spin(10.0, -1e4, 1e4)
            set_button = QPushButton("Set")
            set_button.setFixedWidth(36)
            set_button.clicked.connect(
                lambda _=False, a=axis, lo=lo, hi=hi: self.app.set_range(a, lo.value(), hi.value())
            )
            layout.addWidget(QLabel(axis.upper()), row, 0)
            layout.addWidget(lo, row, 1)
            layout.addWidget(hi, row, 2)
            layout.addWidget(set_button, row, 3)
        auto = QPushButton("Auto Range")
        auto.clicked.connect(self.app.auto_range)
        layout.addWidget(auto, 3, 0, 1, 4)
        return group
