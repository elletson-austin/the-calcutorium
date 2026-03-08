from PySide6.QtWidgets import QWidget, QLineEdit, QHBoxLayout, QPushButton
from PySide6.QtCore import Signal
from .scene import MathFunction

class FunctionEditorWidget(QWidget):
    equation_changed = Signal(MathFunction, str)
    remove_requested = Signal(MathFunction)

    def __init__(self, math_function: MathFunction, parent=None):
        super().__init__(parent)
        self.math_function = math_function

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)

        self.equation_input = QLineEdit()
        self.equation_input.setText(self.math_function.equation_str)
        self.equation_input.returnPressed.connect(self.on_equation_changed)

        remove_btn = QPushButton("×")
        remove_btn.setFixedWidth(24)
        remove_btn.setStyleSheet(
            "QPushButton { color: #FF6666; border: none; font-weight: bold; background: transparent; }"
            "QPushButton:hover { color: #FF0000; }"
        )
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.math_function))

        self.layout.addWidget(self.equation_input)
        self.layout.addWidget(remove_btn)

    def on_equation_changed(self):
        new_equation = self.equation_input.text()
        if new_equation != self.math_function.equation_str:
            self.equation_changed.emit(self.math_function, new_equation)

    def update_equation_text(self):
        self.equation_input.setText(self.math_function.equation_str)
