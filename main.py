from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat # Added import
import sys
from app_window import MainWindow

def main():
    # IMPORTANT: Set the OpenGL surface format BEFORE creating QApplication


    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

