from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget


class TabKeyEventFilter(QObject):
    def __init__(self, render_widget, parent=None):
        super().__init__(parent)
        self.render_widget = render_widget

    def eventFilter(self, watched: QObject, event: QEvent):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab:
            if getattr(self.render_widget, "mouse_hovering", False):
                self.render_widget.keyPressEvent(event)
                return True
        return False


def make_sidebar(functions_panel, simulations_panel, view_panel, output_widget, fixed_width=300):
    """Build the left sidebar with a tab widget and output log at the bottom."""
    sidebar = QWidget()
    sidebar.setFixedWidth(fixed_width)
    sidebar.setStyleSheet("background-color: #2E2E2E;")

    layout = QVBoxLayout(sidebar)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    tabs = QTabWidget()
    tabs.setStyleSheet("""
        QTabWidget::pane { border: 1px solid #555555; background-color: #2E2E2E; }
        QTabBar::tab {
            background-color: #3C3C3C; color: #AAAAAA;
            padding: 6px 10px; border: 1px solid #555555;
            border-bottom: none;
        }
        QTabBar::tab:selected { background-color: #2E2E2E; color: #F0F0F0; }
        QTabBar::tab:hover { background-color: #4A4A4A; color: #F0F0F0; }
    """)
    tabs.addTab(functions_panel, "Functions")
    tabs.addTab(simulations_panel, "Simulations")
    tabs.addTab(view_panel, "View")

    layout.addWidget(tabs, 1)
    layout.addWidget(output_widget)

    return sidebar
