from PySide6.QtWidgets import QFrame, QVBoxLayout, QLineEdit
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class InputWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 200)
        self.setStyleSheet("background-color: #2E2E2E;")
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        layout = QVBoxLayout(self)
        
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter a command...")
        self.command_input.returnPressed.connect(self.on_command)
        self.command_input.setFont(QFont("Consolas", 14))
        self.command_input.setStyleSheet("""
            background-color: #3C3C3C;
            color: #F0F0F0;
            border: 1px solid #555555;
            padding: 5px;
        """)
        
        layout.addWidget(self.command_input)
        layout.addStretch(1) # Pushes the input to the top

    def on_command(self):
        command = self.command_input.text()
        if command =='':
            print("no command entered")
            return
        if command =='snap xy':
            pass
        print(f"Command entered: {command}")

        self.command_input.clear()
