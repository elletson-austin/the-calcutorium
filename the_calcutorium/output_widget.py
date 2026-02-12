from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QFont, QColor, QTextCursor

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

    def write(self, text: str, end: str = '\n'):
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text + end)
        self.moveCursor(QTextCursor.End)

    def write_error(self, text: str, end: str = '\n'):
        self.moveCursor(QTextCursor.End)
        self.setTextColor(QColor("#FF0000"))
        self.insertPlainText(text + end)
        self.moveCursor(QTextCursor.End)
        self.setTextColor(QColor("#F0F0F0"))

