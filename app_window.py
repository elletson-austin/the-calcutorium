from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, QObject, QEvent # QObject and QEvent for event filter
import re

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
        self.input_win = InputWidget()
        left_layout.addWidget(self.input_win)
        self.input_win.command_entered.connect(self.handle_command)

        # Add the left layout and the render widget to the main layout
        main_layout.addLayout(left_layout, 0) # The '0' is the stretch factor
        main_layout.addWidget(self.render_widget, 1) # The '1' makes the render widget take up remaining space

    def handle_command(self, command: str):
        print(f"Command received in MainWindow: {command}")
        
        # Example command: add func "x**3"
        # Example command: remove func "x**2"

        parts = command.split(' ', 2) # Split into at most 3 parts: action, type, function_string
        if len(parts) < 3:
            print("Invalid command format. Expected: <action> <type> \"<function_string>\"")
            return

        action = parts[0].lower()
        type_ = parts[1].lower()
        function_string_quoted = parts[2].strip()

        # Remove quotes from the function string
        match = re.match(r'^\"(.*)\"$', function_string_quoted)
        if not match:
            print("Invalid function string format. Expected: \"<function_string>\"")
            return
        function_string = match.group(1)

        if type_ == "func":
            if action == "add":
                try:
                    # Check if a function with this equation string already exists
                    for obj in self.scene.objects:
                        if isinstance(obj, MathFunction) and obj.equation_str == function_string:
                            print(f"Function '{function_string}' already exists in the scene.")
                            return
                    new_func = MathFunction(function_string)
                    new_func.name = function_string # Assign name for identification
                    self.scene.objects.append(new_func)
                    print(f"Added function: {function_string}")
                except Exception as e:
                    print(f"Error adding function '{function_string}': {e}")
            elif action == "remove":
                func_found = False
                for obj in list(self.scene.objects): # Iterate over a copy to allow modification
                    if isinstance(obj, MathFunction) and obj.equation_str == function_string:
                        self.scene.objects.remove(obj)
                        print(f"Removed function: {function_string}")
                        func_found = True
                        break
                if not func_found:
                    print(f"Function '{function_string}' not found in the scene.")
            else:
                print(f"Unknown action for func type: {action}. Expected 'add' or 'remove'.")
        else:
            print(f"Unknown type: {type_}. Expected 'func'.")



