from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QWidget,
)

from .command_handler import CommandHandler
from .function_editor import FunctionEditorWidget
from .output_widget import OutputWidget
from .panels import FunctionsPanel, SimulationsPanel, ViewPanel
from .render_window import RenderWindow
from .scene import Axes, MathFunction, Scene
from .ui_utils import TabKeyEventFilter, make_sidebar
from .utils import safe_regenerate, sync_function_editors


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("The Calcutorium")
        self.setGeometry(100, 100, 1280, 800)

        self.scene = Scene()
        axes = Axes(length=10.0)
        axes.name = "axes"
        self.scene.objects.append(axes)

        self.render_widget = RenderWindow()
        self.render_widget.set_scene(self.scene)

        self.event_filter = TabKeyEventFilter(self.render_widget)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self.event_filter)

        self.function_editors = {}

        # --- Panels ---
        self.functions_panel = FunctionsPanel()
        self.simulations_panel = SimulationsPanel()
        self.view_panel = ViewPanel()
        self.output_widget = OutputWidget()
        self.output_widget.setFixedHeight(120)
        self.output_widget.write("Welcome to The Calcutorium.")

        sidebar = make_sidebar(
            self.functions_panel,
            self.simulations_panel,
            self.view_panel,
            self.output_widget,
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.render_widget, 1)

        self.render_widget.manual_range_cleared.connect(self._handle_manual_range_cleared)

        self.command_handler = CommandHandler(
            scene=self.scene,
            render_window=self.render_widget,
            output_widget=self.output_widget,
            update_function_editors_callback=self.update_function_editors,
        )

        self._connect_panels()

    def _connect_panels(self):
        # Functions tab
        self.functions_panel.add_function_requested.connect(
            lambda eq: self.command_handler._add_func(eq)
        )

        # Simulations tab
        self.simulations_panel.add_lorenz_requested.connect(self.command_handler._add_lorenz)
        self.simulations_panel.remove_lorenz_requested.connect(self.command_handler._remove_lorenz)
        self.simulations_panel.lorenz_params_changed.connect(self.command_handler.update_lorenz_params)

        self.simulations_panel.add_nbody_requested.connect(self.command_handler._add_nbody)
        self.simulations_panel.remove_nbody_requested.connect(self.command_handler._remove_nbody)
        self.simulations_panel.nbody_params_changed.connect(self.command_handler.update_nbody_params)

        # View tab
        self.view_panel.view_3d_requested.connect(
            lambda: self.command_handler._view_command(["view", "3d"])
        )
        self.view_panel.view_2d_requested.connect(
            lambda plane: self.command_handler._view_2d(plane, self.render_widget.camera)
        )
        self.view_panel.range_set_requested.connect(self._on_range_set)
        self.view_panel.range_auto_requested.connect(
            lambda: self.command_handler._range_command(["range", "auto"])
        )

    def _on_range_set(self, axis: str, min_val: float, max_val: float):
        self.command_handler._range_command(["range", axis, str(min_val), str(max_val)])

    def _handle_manual_range_cleared(self):
        self.output_widget.write("Manual range cleared. Returning to automatic ranging.")

    def update_function_editors(self):
        def factory(func):
            w = FunctionEditorWidget(func)
            w.equation_changed.connect(self.on_equation_changed)
            w.remove_requested.connect(self._on_remove_function)
            return w

        sync_function_editors(
            self.scene,
            self.function_editors,
            self.functions_panel.editors_layout,
            factory,
        )

    def on_equation_changed(self, math_function: MathFunction, new_equation: str):
        self.output_widget.write(f"Equation changed: '{new_equation}'")
        editor = self.function_editors.get(math_function)
        success, err = safe_regenerate(math_function, new_equation)
        if not success:
            self.output_widget.write_error(f"Error: {err}")
            if editor:
                editor.equation_input.setText(getattr(math_function, "equation_str", ""))

    def _on_remove_function(self, math_function: MathFunction):
        if math_function in self.scene.objects:
            self.scene.objects.remove(math_function)
            self.output_widget.write(f"Removed function: {math_function.equation_str}")
            self.update_function_editors()
