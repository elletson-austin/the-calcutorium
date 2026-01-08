import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt, QTimer
import moderngl

class RedOpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctx = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update) # Request repaint
        self.timer.start(16) # ~60 FPS

    def initializeGL(self):
        self.ctx = None

    def resizeGL(self, w: int, h: int):
        if self.ctx:
            self.ctx.viewport = (0, 0, w, h)
        print(f"resizeGL called in test.py: width={w}, height={h}")

    def paintGL(self):
        self.makeCurrent()
        if self.ctx is None:
            self.ctx = moderngl.create_context()
            print("moderngl context created in paintGL in test.py")
        
        if self.ctx:
            self.ctx.clear(1.0, 0.0, 0.0, 1.0) # Clear to red
        print("paintGL called in test.py")

class TestMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Red Window Test")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.gl_widget = RedOpenGLWidget(self)
        layout.addWidget(self.gl_widget)

def main():
    app = QApplication(sys.argv)

    # Crucial: Set the OpenGL surface format BEFORE creating QApplication
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = TestMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
