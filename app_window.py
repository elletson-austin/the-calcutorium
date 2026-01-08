from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from PySide6.QtCore import Qt # Not strictly needed for this file, but often useful
import sys

# Import our custom components
from scene import Scene, Axes
from render_space_pyside import PySideRenderSpace

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("The Calcutorium")
        self.setGeometry(100, 100, 1280, 800) # x, y, width, height

        # Create the Model (Scene)
        self.scene = Scene()
        axes = Axes(length=10.0)
        axes.name = "axes"
        self.scene.objects.append(axes)

        # Create the View (PySideRenderSpace)
        self.render_widget = PySideRenderSpace()
        
        # Connect the View to the Model
        self.render_widget.set_scene(self.scene)

        # Set up the main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0) # Remove margins around the render space
        layout.addWidget(self.render_widget)
