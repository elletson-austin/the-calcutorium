from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea
from PySide6.QtCore import Qt, QObject, QEvent


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


def make_left_panel(input_widget, output_widget, fixed_width=300, bg_color="#2E2E2E"):
    left_panel = QWidget()
    left_panel.setFixedWidth(fixed_width)
    left_panel.setStyleSheet(f"background-color: {bg_color};")

    left_layout = QVBoxLayout(left_panel)
    left_layout.setContentsMargins(5, 5, 5, 5)
    left_layout.setSpacing(5)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll_widget = QWidget()
    function_editors_layout = QVBoxLayout(scroll_widget)
    function_editors_layout.addStretch(1)
    scroll_area.setWidget(scroll_widget)

    left_layout.addWidget(scroll_area, 1)
    left_layout.addWidget(input_widget)
    left_layout.addWidget(output_widget, 1)

    return left_panel, function_editors_layout
