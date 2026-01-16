from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from PySide6.QtCore import Signal, Qt
from scene import MathFunction

class FunctionEditorWidget(QWidget):
    equation_changed = Signal(MathFunction, str)

    def __init__(self, math_function: MathFunction, subscript: int, parent=None):
        super().__init__(parent)
        self.math_function = math_function
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel()
        self.set_subscript(subscript)
        self.equation_input = QLineEdit()
        self.equation_input.setText(self.math_function.equation_str)
        self.equation_input.returnPressed.connect(self.on_equation_changed)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.equation_input)

    def set_subscript(self, subscript: int):
        self.label.setText(f"f<sub>{subscript}</sub>(x) = ")

    def on_equation_changed(self):
        new_equation = self.equation_input.text()
        if new_equation != self.math_function.equation_str:
            self.equation_changed.emit(self.math_function, new_equation)
