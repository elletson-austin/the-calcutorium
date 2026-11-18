"""Single-line command input with history (Up/Down arrows)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import QLineEdit


class CommandBar(QLineEdit):
    command_entered = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Type an equation or command (try: help)")
        self.setFont(QFont("Consolas", 12))
        self.setStyleSheet(
            "background-color: #3C3C3C; color: #F0F0F0; border: 1px solid #555555; padding: 6px;"
        )
        self._history: list[str] = []
        self._history_index = 0
        self.returnPressed.connect(self._submit)

    def _submit(self) -> None:
        text = self.text().strip()
        self.clear()
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._history_index = len(self._history)
        self.command_entered.emit(text)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and self._history:
            step = -1 if event.key() == Qt.Key.Key_Up else 1
            self._history_index = max(0, min(len(self._history), self._history_index + step))
            self.setText(self._history[self._history_index] if self._history_index < len(self._history) else "")
            return
        super().keyPressEvent(event)
