from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QHBoxLayout, QScrollArea
from PySide6.QtCore import Qt, QObject, QEvent # QObject and QEvent for event filter

# Import our custom components
from .scene import Scene, Axes, MathFunction, LorenzAttractor, Grid
from .render_window import RenderWindow
from .camera import Camera2D, Camera3D
from .input_widget import InputWidget
from .function_editor_widget import FunctionEditorWidget
from .output_widget import OutputWidget
from .command_handler import CommandHandler

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
        

        # Create the View (PySideRenderSpace)
        self.render_widget = RenderWindow()
        
        # Connect the View to the Model
        self.render_widget.set_scene(self.scene)

        # Install global event filter for Tab key
        self.event_filter = TabKeyEventFilter(self.render_widget)
        QApplication.instance().installEventFilter(self.event_filter)

        # A dictionary to map MathFunction objects to their editor widgets
        self.function_editors = {}

        # --- Layout Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Left Panel ---
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("background-color: #2E2E2E;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5) # Add spacing between widgets

        # Scroll area for function editors
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        self.function_editors_layout = QVBoxLayout(scroll_widget)
        self.function_editors_layout.addStretch(1)
        scroll_area.setWidget(scroll_widget)

        # Input widget
        self.input_win = InputWidget()
        self.input_win.command_entered.connect(self.handle_command)

        # Output widget
        self.output_widget = OutputWidget()

        self.output_widget.write("hello world")
        
        
        # New layout order and stretches
        left_layout.addWidget(scroll_area, 1)
        left_layout.addWidget(self.input_win)
        left_layout.addWidget(self.output_widget, 1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.render_widget, 1)

        # Connect render_widget signal
        self.render_widget.manual_range_cleared.connect(self._handle_manual_range_cleared)

        # Initialize CommandHandler
        self.command_handler = CommandHandler(
            scene=self.scene,
            render_window=self.render_widget,
            output_widget=self.output_widget,
            update_function_editors_callback=self.update_function_editors
        )

    def _handle_manual_range_cleared(self):
        self.output_widget.write("Manual range cleared due to user interaction. Returning to automatic ranging.")

    def update_function_editors(self):
        # Get an ordered list of MathFunction objects from the scene
        scene_funcs = [obj for obj in self.scene.objects if isinstance(obj, MathFunction)]
        
        # Sync self.function_editors with scene_funcs
        # Remove editors for functions no longer in the scene
        for func_obj in list(self.function_editors.keys()):
            if func_obj not in scene_funcs:
                widget = self.function_editors.pop(func_obj)
                widget.setParent(None)
                widget.deleteLater()

        # Add new editors for functions new to the scene
        for func_obj in scene_funcs:
            if func_obj not in self.function_editors:
                editor_widget = FunctionEditorWidget(func_obj)
                editor_widget.equation_changed.connect(self.on_equation_changed)
                self.function_editors[func_obj] = editor_widget

        # --- Rebuild layout ---
        
        # Detach all editor widgets from the layout
        for widget in self.function_editors.values():
            widget.setParent(None)

        # Re-add widgets in the correct order.
        # Newest functions are at the end of scene_funcs, and should appear at the top of the UI.
        for func_obj in reversed(scene_funcs):
            widget = self.function_editors[func_obj]
            # Insert at the top of the layout to have newest functions on top
            self.function_editors_layout.insertWidget(0, widget)

    def on_equation_changed(self, math_function: MathFunction, new_equation: str):
        self.output_widget.write(f"Equation changed for '{math_function.name}': '{new_equation}'")        
        
        editor = self.function_editors.get(math_function)
        old_equation = math_function.equation_str

        try:
            math_function.regenerate(new_equation)
            # Update name to reflect new equation
            math_function.name = new_equation 
        except ValueError as e:
            self.output_widget.write_error(f"Error regenerating function: {e}")
            # Revert the function and editor to the old state
            try:
                math_function.regenerate(old_equation)
            except ValueError as e:
                self.output_widget.write_error(f"Error reverting function to old state: {e}")
                
            if editor:
                editor.equation_input.setText(old_equation)


    def handle_command(self, command: str):
        self.command_handler.handle_command(command)


