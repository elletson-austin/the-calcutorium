from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, QObject, QEvent # QObject and QEvent for event filter

# Import our custom components
from scene import Scene, Axes, MathFunction, LorenzAttractor
from render_space_pyside import PySideRenderSpace
from input_widget import InputWidget

class TabKeyEventFilter(QObject):
    def __init__(self, render_widget, parent=None):
        super().__init__(parent)
        self.render_widget = render_widget

    def eventFilter(self, watched: QObject, event: QEvent):
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Tab:
            if self.render_widget.mouse_hovering:
                # Manually call the keyPressEvent of the render_widget
                self.render_widget.keyPressEvent(event)
                return True # Event handled, stop propagation
        return False # Event not handled, continue propagation

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

        # Add a math function to the scene
        func = MathFunction("x**2")
        func.name = "y = x^2"
        self.scene.objects.append(func)

        # Add the lorenz attractor to the scene
        lorenz = LorenzAttractor()
        lorenz.name = "Lorenz Attractor"
        self.scene.objects.append(lorenz)

        # Create the View (PySideRenderSpace)
        self.render_widget = PySideRenderSpace()
        
        # Connect the View to the Model
        self.render_widget.set_scene(self.scene)

        # Install global event filter for Tab key
        self.event_filter = TabKeyEventFilter(self.render_widget)
        QApplication.instance().installEventFilter(self.event_filter)

        # --- Layout Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left vertical layout for the input widget
        left_layout = QVBoxLayout()
        left_layout.addStretch(1) # Add a spacer to push the window to the bottom
        
        # Small white window
        input_win = InputWidget()
        left_layout.addWidget(input_win)

        # Add the left layout and the render widget to the main layout
        main_layout.addLayout(left_layout, 0) # The '0' is the stretch factor
        main_layout.addWidget(self.render_widget, 1) # The '1' makes the render widget take up remaining space


