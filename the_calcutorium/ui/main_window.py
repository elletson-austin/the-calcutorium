"""The main window: builds the App and wires every widget to it.

    ┌──────────── sidebar ───────────┬──────────── viewport ────────────┐
    │ [Plots | Simulations | View]   │                                  │
    │                                │        OpenGL rendering          │
    │ output log                     │                                  │
    │                                ├──────────────────────────────────┤
    │                                │ command bar                      │
    └────────────────────────────────┴──────────────────────────────────┘
"""

from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from ..app import App, run_command
from .command_bar import CommandBar
from .output_log import OutputLog
from .panels import PlotsPanel, SimulationsPanel, ViewPanel
from .viewport import TabToggleFilter, Viewport

TAB_STYLE = """
    QTabWidget::pane { border: 1px solid #555555; background-color: #2E2E2E; }
    QTabBar::tab {
        background-color: #3C3C3C; color: #AAAAAA;
        padding: 6px 10px; border: 1px solid #555555; border-bottom: none;
    }
    QTabBar::tab:selected { background-color: #2E2E2E; color: #F0F0F0; }
    QTabBar::tab:hover { background-color: #4A4A4A; color: #F0F0F0; }
"""


class MainWindow(QMainWindow):
    def __init__(self, app: App | None = None):
        super().__init__()
        self.setWindowTitle("The Calcutorium")
        self.resize(1280, 800)
        self.app = app or App()

        self.log = OutputLog()
        self.app.message.subscribe(self.log.write)

        self.viewport = Viewport(self.app)
        self._tab_filter = TabToggleFilter(self.viewport)
        QApplication.instance().installEventFilter(self._tab_filter)

        self.command_bar = CommandBar()
        self.command_bar.command_entered.connect(self._run)

        tabs = QTabWidget()
        tabs.setStyleSheet(TAB_STYLE)
        tabs.addTab(PlotsPanel(self.app), "Plots")
        tabs.addTab(SimulationsPanel(self.app), "Simulations")
        tabs.addTab(ViewPanel(self.app), "View")

        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("background-color: #2E2E2E;")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)
        side.addWidget(tabs, 3)
        side.addWidget(self.log, 2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.viewport, 1)
        right_layout.addWidget(self.command_bar)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(sidebar)
        layout.addWidget(right, 1)
        self.setCentralWidget(central)

        self.app.info("Welcome to The Calcutorium. Type 'help' in the bar below the view.")

    def _run(self, line: str) -> None:
        self.log.write("info", f"> {line}")
        run_command(self.app, line)
