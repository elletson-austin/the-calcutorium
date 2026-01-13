from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QHBoxLayout, QScrollArea
from PySide6.QtCore import Qt, QObject, QEvent # QObject and QEvent for event filter
import shlex
import json

# Import our custom components
from scene import Scene, Axes, MathFunction, LorenzAttractor
from render_space_pyside import PySideRenderSpace
from camera import CameraMode, SnapMode
from input_widget import InputWidget
from function_editor_widget import FunctionEditorWidget
from output_widget import OutputWidget

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
        self.render_widget = PySideRenderSpace(output_widget=self.output_widget)
        
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

        # --- Left Panel ---
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("background-color: #2E2E2E;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

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
        
        left_layout.addWidget(scroll_area)
        left_layout.addWidget(self.input_win)
        left_layout.addWidget(self.output_widget, 1) # Add a stretch factor

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.render_widget, 1)

        self.function_editors = {} # {MathFunction: FunctionEditorWidget}


    def update_function_editors(self):
        # Get current functions from the scene
        scene_funcs = {obj for obj in self.scene.objects if isinstance(obj, MathFunction)}
        
        # Remove widgets for functions that are no longer in the scene
        for func_obj, widget in list(self.function_editors.items()):
            if func_obj not in scene_funcs:
                widget.setParent(None)
                widget.deleteLater()
                del self.function_editors[func_obj]

        # Add widgets for new functions
        for func_obj in scene_funcs:
            if func_obj not in self.function_editors:
                editor_widget = FunctionEditorWidget(func_obj)
                editor_widget.equation_changed.connect(self.on_equation_changed)
                # Insert new editors at the top
                self.function_editors_layout.insertWidget(0, editor_widget)
                self.function_editors[func_obj] = editor_widget

    def on_equation_changed(self, math_function: MathFunction, new_equation: str):
        self.output_widget.append_text(f"Equation changed for '{math_function.name}': '{new_equation}'")        
        try:
            math_function.regenerate(new_equation)
            # Update name to reflect new equation
            math_function.name = new_equation 
        except Exception as e:
            self.output_widget.append_text(f"Error regenerating function: {e}")
            # Optionally, revert the text in the editor if the new equation is invalid
            editor = self.function_editors.get(math_function)
            if editor:
                editor.equation_input.setText(math_function.equation_str)


    def handle_command(self, command: str):
        self.output_widget.append_text(f"Command received in MainWindow: {command}")

        try:
            command_parts = shlex.split(command)
        except ValueError as e:
            self.output_widget.append_text(f"Error parsing command: {e}")
            return
        
        if not command_parts:
            return

        # Handle simple commands first
        if command == "help":
            help_message = """Available commands:
  help - Display this help message
  list - List all objects in the scene
  clear - Clear all objects from the scene except the axes
  save <filename> - Save the current scene to a file
  load <filename> - Load a scene from a file
  view 3d - Switch to 3D view
  view 2d xy - Switch to 2D view on the XY plane
  view 2d xz - Switch to 2D view on the XZ plane
  view 2d yz - Switch to 2D view on the YZ plane
  add lorenz - Add a Lorenz attractor to the scene
  add func \"<function_string>\" - Add a mathematical function to the scene (e.g., 'add func \"x**2\"')
  remove func \"<function_string>\" - Remove a mathematical function from the scene (e.g., 'remove func \"x**2\"')"""
            self.output_widget.append_text(help_message)
            return

        if command == "clear":
            # Remove all objects except for the axes
            self.scene.objects = [obj for obj in self.scene.objects if getattr(obj, 'name', '') == 'axes']
            self.update_function_editors()
            self.output_widget.append_text("Scene cleared.")
            return

        if command == "list":
            if not self.scene.objects:
                self.output_widget.append_text("No objects in the scene.")
                return
            self.output_widget.append_text("Objects in scene:")
            for obj in self.scene.objects:
                self.output_widget.append_text(f"  - {getattr(obj, 'name', 'Unnamed Object')} ({type(obj).__name__})")
            return

        # Handle commands with multiple parts
        action = command_parts[0].lower()

        if action == "save":
            if len(command_parts) < 2:
                self.output_widget.append_text("Invalid 'save' command. Expected: save <filename>")
                return
            filename = command_parts[1]
            try:
                scene_data = self.scene.to_dict()
                with open(filename, 'w') as f:
                    json.dump(scene_data, f, indent=4)
                self.output_widget.append_text(f"Scene saved to {filename}")
            except Exception as e:
                self.output_widget.append_text(f"Error saving scene: {e}")
            return

        if action == "load":
            if len(command_parts) < 2:
                self.output_widget.append_text("Invalid 'load' command. Expected: load <filename>")
                return
            filename = command_parts[1]
            try:
                with open(filename, 'r') as f:
                    scene_data = json.load(f)
                self.scene.from_dict(scene_data)
                self.update_function_editors()
                self.output_widget.append_text(f"Scene loaded from {filename}")
            except FileNotFoundError:
                self.output_widget.append_text(f"Error: File not found '{filename}'")
            except Exception as e:
                self.output_widget.append_text(f"Error loading scene: {e}")
            return

        if action == "view":
            if len(command_parts) < 2:
                self.output_widget.append_text(f"Invalid 'view' command. Expected: 'view <mode> [plane]'. Type 'help' for available commands.")
                return

            mode = command_parts[1].lower()
            if mode == "3d":
                self.render_widget.cam.mode = CameraMode.ThreeD
                self.output_widget.append_text("Switched to 3D Mode")
                return
            
            if mode == "2d":
                if len(command_parts) < 3:
                    self.output_widget.append_text(f"Invalid 'view 2d' command. Expected: 'view 2d <plane>'. Use 'xy', 'xz', or 'yz'.")
                    return
                
                plane = command_parts[2].lower()
                self.render_widget.cam.mode = CameraMode.TwoD
                if plane == "xy":
                    self.render_widget.cam.snap_mode = SnapMode.XY
                    self.output_widget.append_text("Switched to 2D Mode (XY Plane)")
                elif plane == "xz":
                    self.render_widget.cam.snap_mode = SnapMode.XZ
                    self.output_widget.append_text("Switched to 2D Mode (XZ Plane)")
                elif plane == "yz":
                    self.render_widget.cam.snap_mode = SnapMode.YZ
                    self.output_widget.append_text("Switched to 2D Mode (YZ Plane)")
                else:
                    self.output_widget.append_text(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'.")
                return

        if action == "add":
            if len(command_parts) < 2:
                self.output_widget.append_text(f"Invalid 'add' command format: '{command}'. Expected: 'add <type> ...'. Type 'help' for available commands.")
                return

            type_ = command_parts[1].lower()
            if type_ == "lorenz":
                # Check if a Lorenz attractor already exists
                for obj in self.scene.objects:
                    if isinstance(obj, LorenzAttractor):
                        self.output_widget.append_text("Lorenz attractor already exists in the scene.")
                        return
                lorenz = LorenzAttractor()
                lorenz.name = "Lorenz Attractor"
                self.scene.objects.append(lorenz)
                self.output_widget.append_text("Added Lorenz Attractor.")
                return
            
            if type_ == "func":
                if len(command_parts) < 3:
                    self.output_widget.append_text(f"Invalid 'add func' command. Expected: add func \"<value>\".")
                    return
                
                value_string = command_parts[2]

                try:
                    # Check if a function with this equation string already exists
                    for obj in self.scene.objects:
                        if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                            self.output_widget.append_text(f"Function '{value_string}' already exists in the scene.")
                            return
                    new_func = MathFunction(value_string)
                    new_func.name = value_string # Assign name for identification
                    self.scene.objects.append(new_func)
                    self.update_function_editors()
                    self.output_widget.append_text(f"Added function: {value_string}")
                except Exception as e:
                    self.output_widget.append_text(f"Error adding function '{value_string}': {e}")
                return

        if action == "remove":
            if len(command_parts) < 2:
                self.output_widget.append_text(f"Invalid 'remove' command format: '{command}'. Expected: 'remove <type> \"<value>\"'. Type 'help' for available commands.")
                return
            
            type_ = command_parts[1].lower()
            if type_ == "func":
                if len(command_parts) < 3:
                    self.output_widget.append_text(f"Invalid 'remove func' command. Expected: remove func \"<value>\".")
                    return

                value_string = command_parts[2]

                func_found = False
                for obj in list(self.scene.objects): # Iterate over a copy to allow modification
                    if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                        self.scene.objects.remove(obj)
                        self.update_function_editors()
                        self.output_widget.append_text(f"Removed function: {value_string}")
                        func_found = True
                        break
                self.output_widget.append_text(f"Function '{value_string}' not found in the scene.")
                return

        self.output_widget.append_text(f"Unknown or invalid command: '{command}'. Type 'help' for available commands.")
        return
