from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QHBoxLayout
from PySide6.QtCore import Qt

# Import our custom components
from .scene import Scene, Axes, MathFunction
from .render_window import RenderWindow
from .input_widget import InputWidget
from .function_editor_widget import FunctionEditorWidget
from .output_widget import OutputWidget
from .command_handler import CommandHandler
from .ui_utils import TabKeyEventFilter, make_left_panel
from .utils import safe_regenerate, sync_function_editors


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
        

        self.render_widget = RenderWindow()
        self.render_widget.set_scene(self.scene)

        # Install global event filter for Tab key
        self.event_filter = TabKeyEventFilter(self.render_widget)
        QApplication.instance().installEventFilter(self.event_filter)

        self.function_editors = {}

        # --- Layout Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Input and output widgets
        self.input_win = InputWidget()
        self.input_win.command_entered.connect(self.handle_command)
        self.output_widget = OutputWidget()
        self.output_widget.write("hello world")

        left_panel, self.function_editors_layout = make_left_panel(self.input_win, self.output_widget)
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
        def factory(func):
            w = FunctionEditorWidget(func)
            w.equation_changed.connect(self.on_equation_changed)
            return w

        sync_function_editors(self.scene, self.function_editors, self.function_editors_layout, factory)

    def on_equation_changed(self, math_function: MathFunction, new_equation: str):
        self.output_widget.write(f"Equation changed for '{math_function.name}': '{new_equation}'")

        editor = self.function_editors.get(math_function)
        success, err = safe_regenerate(math_function, new_equation)
        if not success:
            self.output_widget.write_error(f"Error regenerating function: {err}")
            if editor:
                editor.equation_input.setText(getattr(math_function, "equation_str", ""))


    def handle_command(self, command: str):
        self.command_handler.handle_command(command)


