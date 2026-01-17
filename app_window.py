from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QHBoxLayout, QScrollArea
from PySide6.QtCore import Qt, QObject, QEvent # QObject and QEvent for event filter
import shlex
import json
import sys

# Import our custom components
from scene import Scene, Axes, MathFunction, LorenzAttractor, Grid
from rendering import RenderSpace, CameraMode, SnapMode
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
        
        grid = Grid()
        grid.name = "grid"
        self.scene.objects.append(grid)

        # Create the View (PySideRenderSpace)
        self.render_widget = RenderSpace()
        
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
                # The subscript passed here is a placeholder, it will be correctly set below
                editor_widget = FunctionEditorWidget(func_obj, 0)
                editor_widget.equation_changed.connect(self.on_equation_changed)
                self.function_editors[func_obj] = editor_widget

        # --- Rebuild layout and update subscripts ---
        
        # Detach all editor widgets from the layout
        for widget in self.function_editors.values():
            widget.setParent(None)

        # Re-add widgets in the correct order with updated subscripts.
        # Newest functions are at the end of scene_funcs, and should appear at the top of the UI.
        for i, func_obj in enumerate(reversed(scene_funcs)):
            subscript = len(scene_funcs) - i
            widget = self.function_editors[func_obj]
            widget.set_subscript(subscript)
            # Insert at the top of the layout to have newest functions on top
            self.function_editors_layout.insertWidget(0, widget)

    def on_equation_changed(self, math_function: MathFunction, new_equation: str):
        self.output_widget.write(f"Equation changed for '{math_function.name}': '{new_equation}'")        
        try:
            math_function.regenerate(new_equation)
            # Update name to reflect new equation
            math_function.name = new_equation 
        except Exception as e:
            self.output_widget.write_error(f"Error regenerating function: {e}")
            # Optionally, revert the text in the editor if the new equation is invalid
            editor = self.function_editors.get(math_function)
            if editor:
                editor.equation_input.setText(math_function.equation_str)


    def handle_command(self, command: str):
        self.output_widget.write(f"Command received in MainWindow: {command}")

        try:
            command_parts = shlex.split(command)
        except ValueError as e:
            self.output_widget.write_error(f"Error parsing command: {e}")
            return
        
        if not command_parts:
            return

        # Handle simple commands first
        if command == "help":
            help_message = """Available commands:
  help - Display this help message
  list - List all objects in the scene, with detailed information for functions
  clear - Clear all objects from the scene except the axes
  save <filename> - Save the current scene to a file
  load <filename> - Load a scene from a file
  view 3d - Switch to 3D view
  view 2d xy - Switch to 2D view on the XY plane
  view 2d xz - Switch to 2D view on the XZ plane
  view 2d yz - Switch to 2D view on the YZ plane
  range <x|y|z> <min> <max> - Manually set the visible range for an axis (e.g., 'range x -10 10')
  range auto - Reset all axes to automatic ranging
  add lorenz - Add a Lorenz attractor to the scene
  add func \"<function_string>\" - Add a mathematical function to the scene (e.g., 'add func \"x**2\"')
  remove func \"<function_string>\" - Remove a mathematical function from the scene (e.g., 'remove func \"x**2\"')"""
            self.output_widget.write(help_message)
            return

        if command == "clear":
            # Remove all objects except for the axes and grid
            self.scene.objects = [obj for obj in self.scene.objects if getattr(obj, 'name', '') in ['axes', 'grid']]
            self.update_function_editors()
            self.output_widget.write("Scene cleared.")
            return

        if command == "list":
            if not self.scene.objects:
                self.output_widget.write("No objects in the scene.")
                return
            
            self.output_widget.write("Objects in scene:")
            
            # Separate functions from other objects for display
            functions = [obj for obj in self.scene.objects if isinstance(obj, MathFunction)]
            other_objects = [obj for obj in self.scene.objects if not isinstance(obj, MathFunction)]

            if functions:
                self.output_widget.write("  --- Functions ---")
                for i, func in enumerate(functions):
                    self.output_widget.write(f"    [{i}] Name: {func.name}, Equation: '{func.equation_str}'")
            
            if other_objects:
                self.output_widget.write("  --- Other Objects ---")
                for obj in other_objects:
                    self.output_widget.write(f"    Name: {getattr(obj, 'name', 'Unnamed Object')} ({type(obj).__name__})")
            return

        # Handle commands with multiple parts
        action = command_parts[0].lower()

        if action == "range":
            if self.render_widget.cam.mode != CameraMode.TwoD:
                self.output_widget.write_error("The 'range' command is only available in 2D view modes. Use 'view 2d ...' first.")
                return

            if len(command_parts) == 2 and command_parts[1].lower() == 'auto':
                self.render_widget.manual_ranges.clear()
                self.output_widget.write("Switched to automatic ranging.")
                return

            if len(command_parts) != 4:
                self.output_widget.write_error("Invalid 'range' command. Use: 'range <x|y|z> <min> <max>' or 'range auto'.")
                return
            
            axis_str = command_parts[1].lower()
            if axis_str not in ['x', 'y', 'z']:
                self.output_widget.write_error(f"Invalid axis '{axis_str}'. Use 'x', 'y', or 'z'.")
                return

            try:
                min_val, max_val = float(command_parts[2]), float(command_parts[3])
            except ValueError:
                self.output_widget.write_error("Invalid range values. Min and max must be numbers.")
                return
            
            if min_val >= max_val:
                self.output_widget.write_error("Min range value must be less than max value.")
                return

            self.render_widget.manual_ranges[axis_str] = (min_val, max_val)
            self.output_widget.write(f"Set manual range for {axis_str}-axis to ({min_val}, {max_val}).")

            # Adjust camera center and distance if both relevant ranges are now manually set
            current_snap_mode = self.render_widget.cam.snap_mode
            h_axis, v_axis = None, None

            if current_snap_mode == SnapMode.XY:
                h_axis, v_axis = 'x', 'y'
            elif current_snap_mode == SnapMode.XZ:
                h_axis, v_axis = 'x', 'z'
            elif current_snap_mode == SnapMode.YZ:
                h_axis, v_axis = 'z', 'y' # Z is horizontal, Y is vertical

            if h_axis and v_axis and h_axis in self.render_widget.manual_ranges and v_axis in self.render_widget.manual_ranges:
                h_range = self.render_widget.manual_ranges[h_axis]
                v_range = self.render_widget.manual_ranges[v_axis]

                # Update camera center
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                self.render_widget.cam.position_center[axis_map[h_axis]] = (h_range[0] + h_range[1]) / 2
                self.render_widget.cam.position_center[axis_map[v_axis]] = (v_range[0] + v_range[1]) / 2
                
                range_width = h_range[1] - h_range[0]
                range_height = v_range[1] - v_range[0]
                
                if range_width <= 0 or range_height <= 0:
                    self.output_widget.write_error("Calculated range width or height is non-positive. Cannot adjust camera.")
                    return
                
                range_aspect = range_width / range_height
                
                width, height = self.render_widget.width(), self.render_widget.height()
                if height == 0:
                    self.output_widget.write_error("Render widget height is zero. Cannot adjust camera aspect.")
                    return
                window_aspect = width / height
                
                # Adjust camera distance to fit the range within the window, maintaining chosen aspect
                if window_aspect > range_aspect:
                    # Window is wider than the desired range, fit to height
                    self.render_widget.cam.distance = range_height
                else:
                    # Window is taller or equal aspect, fit to width
                    self.render_widget.cam.distance = range_width / window_aspect
                
                self.output_widget.write(f"Camera adjusted to fit manual ranges for {h_axis}/{v_axis} plane.")
            return

        if action == "save":
            if len(command_parts) < 2:
                self.output_widget.write_error("Invalid 'save' command. Expected: save <filename>")
                return
            filename = command_parts[1]
            try:
                scene_data = self.scene.to_dict()
                with open(filename, 'w') as f:
                    json.dump(scene_data, f, indent=4)
                self.output_widget.write(f"Scene saved to {filename}")
            except Exception as e:
                self.output_widget.write_error(f"Error saving scene: {e}")
            return

        if action == "load":
            if len(command_parts) < 2:
                self.output_widget.write_error("Invalid 'load' command. Expected: load <filename>")
                return
            filename = command_parts[1]
            try:
                with open(filename, 'r') as f:
                    scene_data = json.load(f)
                self.scene.from_dict(scene_data)
                self.update_function_editors()
                self.output_widget.write(f"Scene loaded from {filename}")
            except FileNotFoundError:
                self.output_widget.write_error(f"Error: File not found '{filename}'")
            except Exception as e:
                self.output_widget.write_error(f"Error loading scene: {e}")
            return

        if action == "view":
            if len(command_parts) < 2:
                self.output_widget.write_error(f"Invalid 'view' command. Expected: 'view <mode> [plane]'. Type 'help' for available commands.")
                return

            mode = command_parts[1].lower()
            if mode == "3d":
                self.render_widget.cam.mode = CameraMode.ThreeD
                self.output_widget.write("Switched to 3D Mode")
                return
            
            if mode == "2d":
                if len(command_parts) < 3:
                    self.output_widget.write_error(f"Invalid 'view 2d' command. Expected: 'view 2d <plane>'. Use 'xy', 'xz', or 'yz'.")
                    return
                
                plane = command_parts[2].lower()
                self.render_widget.cam.mode = CameraMode.TwoD
                if plane == "xy":
                    self.render_widget.cam.snap_mode = SnapMode.XY
                    self.output_widget.write("Switched to 2D Mode (XY Plane)")
                elif plane == "xz":
                    self.render_widget.cam.snap_mode = SnapMode.XZ
                    self.output_widget.write("Switched to 2D Mode (XZ Plane)")
                elif plane == "yz":
                    self.render_widget.cam.snap_mode = SnapMode.YZ
                    self.output_widget.write("Switched to 2D Mode (YZ Plane)")
                else:
                    self.output_widget.write_error(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'.")
                return

        if action == "add":
            if len(command_parts) < 2:
                self.output_widget.write_error(f"Invalid 'add' command format: '{command}'. Expected: 'add <type> ...'. Type 'help' for available commands.")
                return

            type_ = command_parts[1].lower()
            if type_ == "lorenz":
                # Check if a Lorenz attractor already exists
                for obj in self.scene.objects:
                    if isinstance(obj, LorenzAttractor):
                        self.output_widget.write("Lorenz attractor already exists in the scene.")
                        return
                lorenz = LorenzAttractor()
                lorenz.name = "Lorenz Attractor"
                self.scene.objects.append(lorenz)
                self.output_widget.write("Added Lorenz Attractor.")
                return
            
            if type_ == "func":
                if len(command_parts) < 3:
                    self.output_widget.write_error(f"Invalid 'add func' command. Expected: add func \"<value>\".")
                    return
                
                value_string = command_parts[2]

                try:
                    # Check if a function with this equation string already exists
                    for obj in self.scene.objects:
                        if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                            self.output_widget.write(f"Function '{value_string}' already exists in the scene.")
                            return
                    new_func = MathFunction(value_string)
                    new_func.name = value_string # Assign name for identification
                    self.scene.objects.append(new_func)
                    self.update_function_editors()
                    self.output_widget.write(f"Added function: {value_string}")
                except Exception as e:
                    self.output_widget.write_error(f"Error adding function '{value_string}': {e}")
                return

        if action == "remove":
            if len(command_parts) < 2:
                self.output_widget.write_error(f"Invalid 'remove' command format: '{command}'. Expected: 'remove <type> \"<value>\"'. Type 'help' for available commands.")
                return
            
            type_ = command_parts[1].lower()
            if type_ == "func":
                if len(command_parts) < 3:
                    self.output_widget.write_error(f"Invalid 'remove func' command. Expected: remove func \"<value>\".")
                    return

                value_string = command_parts[2]

                func_to_remove = None
                for obj in self.scene.objects:
                    if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                        func_to_remove = obj
                        break
                
                if func_to_remove:
                    self.scene.objects.remove(func_to_remove)
                    self.update_function_editors()
                    self.output_widget.write(f"Removed function: {value_string}")
                else:
                    self.output_widget.write(f"Function '{value_string}' not found in the scene.")
                return

        self.output_widget.write(f"Unknown or invalid command: '{command}'. Type 'help' for available commands.")
        return

