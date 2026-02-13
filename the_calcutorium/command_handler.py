import shlex
import json
from typing import TYPE_CHECKING, Dict, Callable, List

if TYPE_CHECKING:
    from .scene import Scene, MathFunction, LorenzAttractor
    from .rendering import RenderWindow, SnapMode, Camera2D, Camera3D
    from .output_widget import OutputWidget

class CommandHandler:
    def __init__(self, scene: 'Scene', render_window: 'RenderWindow', output_widget: 'OutputWidget', update_function_editors_callback: Callable):
        self.scene = scene
        self.render_window = render_window
        self.output_widget = output_widget
        self.update_function_editors_callback = update_function_editors_callback # Callback to update UI in MainWindow

        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "help": self._help_command,
            "list": self._list_command,
            "clear": self._clear_command,
            "save": self._save_command,
            "load": self._load_command,
            "view": self._view_command,
            "range": self._range_command,
            "add": self._add_command,
            "remove": self._remove_command,
        }

    def handle_command(self, command: str):
        self.output_widget.write(f"Command received: {command}")

        try:
            command_parts = shlex.split(command)
        except ValueError as e:
            self.output_widget.write_error(f"Error parsing command: {e}")
            return
        
        if not command_parts:
            return

        command_name = command_parts[0].lower()
        handler = self.commands.get(command_name)

        if handler:
            try:
                handler(command_parts)
            except Exception as e:
                self.output_widget.write_error(f"Error executing command '{command_name}': {e}")
        else:
            self.output_widget.write(f"Unknown or invalid command: '{command}'. Type 'help' for available commands.")

    def _help_command(self, command_parts: list[str]):
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
  add func "<function_string>" - Add a mathematical function to the scene (e.g., 'add func "x**2"')
  remove func "<function_string>" - Remove a mathematical function from the scene (e.g., 'remove func "x**2"')"""
        self.output_widget.write(help_message)

    def _clear_command(self, command_parts: list[str]):
        from .scene import Axes, Grid # Import here to avoid circular dependency with TYPE_CHECKING
        self.scene.objects = [obj for obj in self.scene.objects if isinstance(obj, (Axes, Grid))]
        self.update_function_editors_callback()
        self.output_widget.write("Scene cleared.")

    def _list_command(self, command_parts: list[str]):
        from .scene import MathFunction # Import here for type checking
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

    def _save_command(self, command_parts: list[str]):
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

    def _load_command(self, command_parts: list[str]):
        if len(command_parts) < 2:
            self.output_widget.write_error("Invalid 'load' command. Expected: load <filename>")
            return
        filename = command_parts[1]
        try:
            with open(filename, 'r') as f:
                scene_data = json.load(f)
            self.scene.from_dict(scene_data)
            self.update_function_editors_callback()
            self.output_widget.write(f"Scene loaded from {filename}")
        except FileNotFoundError:
            self.output_widget.write_error(f"Error: File not found '{filename}'")
        except Exception as e:
            self.output_widget.write_error(f"Error loading scene: {e}")

    def _view_command(self, command_parts: list[str]):
        from .rendering import Camera3D, Camera2D, SnapMode # Import here for type checking
        if len(command_parts) < 2:
            self.output_widget.write_error("Invalid 'view' command. Expected: 'view <mode> [plane]'. Type 'help' for available commands.")
            return

        mode = command_parts[1].lower()
        current_cam = self.render_window.camera
        
        if mode == "3d":
            if isinstance(current_cam, Camera3D):
                self.output_widget.write("Already in 3D Mode.")
                return
            new_cam = Camera3D(position_center=current_cam.position_center, distance=current_cam.distance)
            self.render_window.set_camera(new_cam)
            self.output_widget.write("Switched to 3D Mode")
            return
        
        if mode == "2d":
            if len(command_parts) < 3:
                self.output_widget.write_error("Invalid 'view 2d' command. Expected: 'view 2d <plane>'. Use 'xy', 'xz', or 'yz'.")
                return
            
            plane = command_parts[2].lower()
            snap_map = {"xy": SnapMode.XY, "xz": SnapMode.XZ, "yz": SnapMode.YZ}
            snap_mode = snap_map.get(plane)

            if snap_mode:
                if isinstance(current_cam, Camera2D) and current_cam.snap_mode == snap_mode:
                    self.output_widget.write(f"Already in 2D Mode ({plane.upper()} Plane).")
                    return
                new_cam = Camera2D(
                    position_center=current_cam.position_center,
                    distance=current_cam.distance,
                    snap_mode=snap_mode
                )
                self.render_window.set_camera(new_cam)
                self.output_widget.write(f"Switched to 2D Mode ({plane.upper()} Plane)")
            else:
                self.output_widget.write_error(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'.")
            return

    def _range_command(self, command_parts: list[str]):
        from .rendering import Camera2D, SnapMode # Import here for type checking
        if not isinstance(self.render_window.camera, Camera2D):
            self.output_widget.write_error("The 'range' command is only available in 2D view modes. Use 'view 2d ...' first.")
            return

        if len(command_parts) == 2 and command_parts[1].lower() == 'auto':
            self.render_window.manual_ranges.clear()
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

        self.render_window.manual_ranges[axis_str] = (min_val, max_val)
        self.output_widget.write(f"Set manual range for {axis_str}-axis to ({min_val}, {max_val}).")

        # Adjust camera center and distance if both relevant ranges are now manually set
        current_snap_mode = self.render_window.camera.snap_mode
        h_axis, v_axis = None, None

        if current_snap_mode == SnapMode.XY:
            h_axis, v_axis = 'x', 'y'
        elif current_snap_mode == SnapMode.XZ:
            h_axis, v_axis = 'x', 'z'
        elif current_snap_mode == SnapMode.YZ:
            h_axis, v_axis = 'z', 'y' # Z is horizontal, Y is vertical

        if h_axis and v_axis and h_axis in self.render_window.manual_ranges and v_axis in self.render_window.manual_ranges:
            h_range = self.render_window.manual_ranges[h_axis]
            v_range = self.render_window.manual_ranges[v_axis]

            # Update camera center
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            self.render_window.camera.position_center[axis_map[h_axis]] = (h_range[0] + h_range[1]) / 2
            self.render_window.camera.position_center[axis_map[v_axis]] = (v_range[0] + v_range[1]) / 2
            
            range_width = h_range[1] - h_range[0]
            range_height = v_range[1] - v_range[0]
            
            if range_width <= 0 or range_height <= 0:
                self.output_widget.write_error("Calculated range width or height is non-positive. Cannot adjust camera.")
                return
            
            range_aspect = range_width / range_height
            
            width, height = self.render_window.width(), self.render_window.height()
            if height == 0:
                self.output_widget.write_error("Render widget height is zero. Cannot adjust camera aspect.")
                return
            window_aspect = width / height
            
            # Adjust camera distance to fit the range within the window, maintaining chosen aspect
            if window_aspect > range_aspect:
                # Window is wider than the desired range, fit to height
                self.render_window.camera.distance = range_height
            else:
                # Window is taller or equal aspect, fit to width
                self.render_window.camera.distance = range_width / window_aspect
            
            self.output_widget.write(f"Camera adjusted to fit manual ranges for {h_axis}/{v_axis} plane.")

    def _add_command(self, command_parts: list[str]):
        from .scene import MathFunction, LorenzAttractor # Import here for type checking
        if len(command_parts) < 2:
            self.output_widget.write_error(f"Invalid 'add' command format. Expected: 'add <type> ...'. Type 'help' for available commands.")
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
            self.update_function_editors_callback() # Lorenz is not a function, but might need editor update
            return
        
        if type_ == "func":
            if len(command_parts) < 3:
                self.output_widget.write_error(f"Invalid 'add func' command. Expected: add func "<value>".")
                return
            
            value_string = command_parts[2]

            try:
                # Check if a function with this equation string already exists
                for obj in self.scene.objects:
                    if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                        self.output_widget.write(f"Function '{value_string}' already exists in the scene.")
                        return
                new_func = MathFunction(value_string, output_widget=self.output_widget) # Pass output_widget
                new_func.name = value_string # Assign name for identification
                self.scene.objects.append(new_func)
                self.update_function_editors_callback()
                self.output_widget.write(f"Added function: {value_string}")
            except Exception as e:
                self.output_widget.write_error(f"Error adding function '{value_string}': {e}")
            return
    
    def _remove_command(self, command_parts: list[str]):
        from .scene import MathFunction # Import here for type checking
        if len(command_parts) < 2:
            self.output_widget.write_error(f"Invalid 'remove' command format. Expected: 'remove <type> "<value>"'. Type 'help' for available commands.")
            return
        
        type_ = command_parts[1].lower()
        if type_ == "func":
            if len(command_parts) < 3:
                self.output_widget.write_error(f"Invalid 'remove func' command. Expected: remove func "<value>".")
                return

            value_string = command_parts[2]

            func_to_remove = None
            for obj in self.scene.objects:
                if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                    func_to_remove = obj
                    break
            
            if func_to_remove:
                self.scene.objects.remove(func_to_remove)
                self.update_function_editors_callback()
                self.output_widget.write(f"Removed function: {value_string}")
            else:
                self.output_widget.write(f"Function '{value_string}' not found in the scene.")
            return
