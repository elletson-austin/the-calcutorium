from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QFont

class OutputWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("""
            background-color: #2E2E2E;
            color: #F0F0F0;
            border: 1px solid #555555;
            padding: 5px;
        """)

    def append_text(self, text):
        self.moveCursor(self.textCursor().End)
        self.insertPlainText(text + '\\n')
