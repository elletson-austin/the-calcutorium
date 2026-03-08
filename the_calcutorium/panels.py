from PySide6.QtCore import Qt, Signal
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


class FunctionsPanel(QWidget):
    add_function_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(PANEL_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        add_row = QHBoxLayout()
        self.eq_input = QLineEdit()
        self.eq_input.setPlaceholderText("e.g. y = x**2")
        self.eq_input.returnPressed.connect(self._on_add)
        add_btn = QPushButton("Add")
        add_btn.setFixedWidth(50)
        add_btn.clicked.connect(self._on_add)
        add_row.addWidget(self.eq_input)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editors_container = QWidget()
        self.editors_layout = QVBoxLayout(editors_container)
        self.editors_layout.setContentsMargins(0, 0, 0, 0)
        self.editors_layout.setSpacing(4)
        self.editors_layout.addStretch(1)
        scroll.setWidget(editors_container)
        layout.addWidget(scroll, 1)

    def _on_add(self):
        eq = self.eq_input.text().strip()
        if eq:
            self.add_function_requested.emit(eq)
            self.eq_input.clear()


class SimulationsPanel(QWidget):
    add_lorenz_requested = Signal()
    remove_lorenz_requested = Signal()
    lorenz_params_changed = Signal(dict)

    add_nbody_requested = Signal()
    remove_nbody_requested = Signal()
    nbody_params_changed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(PANEL_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._build_lorenz_group())
        layout.addWidget(self._build_nbody_group())
        layout.addStretch(1)

    def _build_lorenz_group(self):
        group = QGroupBox("Lorenz Attractor")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        add_btn.clicked.connect(self.add_lorenz_requested)
        remove_btn.clicked.connect(self.remove_lorenz_requested)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        layout.addLayout(btn_row)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.lorenz_sigma = self._spin(10.0, 0.1, 100.0)
        self.lorenz_rho   = self._spin(28.0, 0.1, 100.0)
        self.lorenz_beta  = self._spin(8.0 / 3.0, 0.01, 10.0)
        self.lorenz_dt    = self._spin(0.001, 0.0001, 0.1, decimals=4)
        self.lorenz_steps = self._int_spin(5, 1, 50)

        for row, (label, widget) in enumerate([
            ("sigma", self.lorenz_sigma),
            ("rho",   self.lorenz_rho),
            ("beta",  self.lorenz_beta),
            ("dt",    self.lorenz_dt),
            ("steps", self.lorenz_steps),
        ]):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_lorenz_apply)
        layout.addWidget(apply_btn)
        return group

    def _build_nbody_group(self):
        group = QGroupBox("N-Body Simulation")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        add_btn.clicked.connect(self.add_nbody_requested)
        remove_btn.clicked.connect(self.remove_nbody_requested)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        layout.addLayout(btn_row)

        grid = QGridLayout()
        grid.setSpacing(4)
        self.nbody_bodies    = self._int_spin(4000, 100, 50000)
        self.nbody_G         = self._spin(1.0, 0.01, 100.0)
        self.nbody_dt        = self._spin(0.01, 0.001, 1.0, decimals=3)
        self.nbody_softening = self._spin(1.0, 0.01, 10.0)
        self.nbody_steps     = self._int_spin(5, 1, 50)

        for row, (label, widget) in enumerate([
            ("bodies",    self.nbody_bodies),
            ("G",         self.nbody_G),
            ("dt",        self.nbody_dt),
            ("softening", self.nbody_softening),
            ("steps",     self.nbody_steps),
        ]):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addLayout(grid)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_nbody_apply)
        layout.addWidget(apply_btn)
        return group

    def _spin(self, value, min_val, max_val, decimals=2):
        s = QDoubleSpinBox()
        s.setRange(min_val, max_val)
        s.setValue(value)
        s.setDecimals(decimals)
        return s

    def _int_spin(self, value, min_val, max_val):
        s = QSpinBox()
        s.setRange(min_val, max_val)
        s.setValue(value)
        return s

    def _on_lorenz_apply(self):
        self.lorenz_params_changed.emit({
            "sigma": self.lorenz_sigma.value(),
            "rho":   self.lorenz_rho.value(),
            "beta":  self.lorenz_beta.value(),
            "dt":    self.lorenz_dt.value(),
            "steps": self.lorenz_steps.value(),
        })

    def _on_nbody_apply(self):
        self.nbody_params_changed.emit({
            "dt":        self.nbody_dt.value(),
            "G":         self.nbody_G.value(),
            "softening": self.nbody_softening.value(),
            "num_bodies": self.nbody_bodies.value(),
            "steps":     self.nbody_steps.value(),
        })


class ViewPanel(QWidget):
    view_3d_requested  = Signal()
    view_2d_requested  = Signal(str)   # plane: 'xy', 'xz', 'yz'
    range_set_requested = Signal(str, float, float)  # axis, min, max
    range_auto_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(PANEL_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._build_view_mode_group())
        layout.addWidget(self._build_range_group())
        layout.addStretch(1)

    def _build_view_mode_group(self):
        group = QGroupBox("View Mode")
        layout = QGridLayout(group)
        layout.setSpacing(4)

        btn_3d = QPushButton("3D")
        btn_xy = QPushButton("2D XY")
        btn_xz = QPushButton("2D XZ")
        btn_yz = QPushButton("2D YZ")

        btn_3d.clicked.connect(self.view_3d_requested)
        btn_xy.clicked.connect(lambda: self.view_2d_requested.emit("xy"))
        btn_xz.clicked.connect(lambda: self.view_2d_requested.emit("xz"))
        btn_yz.clicked.connect(lambda: self.view_2d_requested.emit("yz"))

        layout.addWidget(btn_3d, 0, 0, 1, 2)
        layout.addWidget(btn_xy, 1, 0)
        layout.addWidget(btn_xz, 1, 1)
        layout.addWidget(btn_yz, 2, 0)
        return group

    def _build_range_group(self):
        group = QGroupBox("Range (2D)")
        layout = QGridLayout(group)
        layout.setSpacing(4)

        for row, axis in enumerate(["x", "y", "z"]):
            mn = QDoubleSpinBox()
            mx = QDoubleSpinBox()
            for spin in (mn, mx):
                spin.setRange(-10000, 10000)
                spin.setDecimals(2)
            mn.setValue(-10.0)
            mx.setValue(10.0)

            set_btn = QPushButton("Set")
            set_btn.setFixedWidth(36)
            set_btn.clicked.connect(
                lambda _, a=axis, lo=mn, hi=mx:
                self.range_set_requested.emit(a, lo.value(), hi.value())
            )

            layout.addWidget(QLabel(axis.upper()), row, 0)
            layout.addWidget(mn, row, 1)
            layout.addWidget(mx, row, 2)
            layout.addWidget(set_btn, row, 3)

        auto_btn = QPushButton("Auto Range")
        auto_btn.clicked.connect(self.range_auto_requested)
        layout.addWidget(auto_btn, 3, 0, 1, 4)
        return group
