from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel, QHBoxLayout
from PySide6.QtCore import Signal, Qt
from .scene import MathFunction

class FunctionEditorWidget(QWidget):
    equation_changed = Signal(MathFunction, str)

    def __init__(self, math_function: MathFunction, parent=None):
        super().__init__(parent)
        self.math_function = math_function
        
        # A horizontal layout for the label and the equation input
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)

        self.update_label()

        self.equation_input = QLineEdit()
        self.equation_input.setText(self.math_function.equation_str)
        self.equation_input.returnPressed.connect(self.on_equation_changed)
        
        self.layout.addWidget(self.equation_input)

    def on_equation_changed(self):
        new_equation = self.equation_input.text()
        if new_equation != self.math_function.equation_str:
            self.equation_changed.emit(self.math_function, new_equation)

    def update_equation_text(self):
        self.equation_input.setText(self.math_function.equation_str)
