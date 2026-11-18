"""Read-only log of what the app did. Errors are red."""

from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import QTextEdit

_COLORS = {"info": QColor("#F0F0F0"), "error": QColor("#FF6B6B")}


class OutputLog(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet(
            "background-color: #262626; color: #F0F0F0; border: 1px solid #555555; padding: 5px;"
        )

    def write(self, level: str, text: str) -> None:
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.setTextColor(_COLORS.get(level, _COLORS["info"]))
        self.insertPlainText(text + "\n")
        self.moveCursor(QTextCursor.MoveOperation.End)
