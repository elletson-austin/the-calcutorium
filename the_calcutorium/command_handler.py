import shlex
from typing import TYPE_CHECKING, Callable, Dict, List

from .scene import Grid

if TYPE_CHECKING:
    from .output_widget import OutputWidget
    from .render_window import RenderWindow
    from .scene import Scene


class CommandHandler:
    def __init__(
        self,
        scene: "Scene",
        render_window: "RenderWindow",
        output_widget: "OutputWidget",
        update_function_editors_callback: Callable,
    ):
        self.scene = scene
        self.render_window = render_window
        self.output_widget = output_widget
        self.update_function_editors_callback = (
            update_function_editors_callback  # Callback to update UI in MainWindow
        )

        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "help": self._help_command,
            "clear": self._clear_command,
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
                self.output_widget.write_error(
                    f"Error executing command '{command_name}': {e}"
                )
        else:
            self.output_widget.write(
                f"Unknown or invalid command: '{command}'. Type 'help' for available commands."
            )

    def _help_command(self, command_parts: list[str]):
        help_message = """Available commands:
  help - Display this help message
  clear - Clear all objects from the scene except the axes
  view 3d - Switch to 3D view
  view 2d <plane> - Switch to 2D view on the specified plane (e.g., 'view 2d xy', 'view 2d xz', 'view 2d yz')
  range <axis> <min> <max> - Manually set the visible range for an axis (e.g., 'range x -10 10')
  range auto - Reset all axes to automatic ranging
  add <type> - Add an object to the scene (e.g., 'add lorenz', 'add nbody', 'add func "x**2"')
  remove <type> <name> - Remove an object from the scene (e.g., 'remove lorenz', 'remove nbody', 'remove func "x**2"')"""
        self.output_widget.write(help_message)

    def _clear_command(self, command_parts: list[str]):
        from .scene import Axes, Grid

        self.scene.objects = [
            obj for obj in self.scene.objects if isinstance(obj, (Axes, Grid))
        ]
        self.update_function_editors_callback()
        self.output_widget.write("Scene cleared.")

    def _view_command(self, command_parts: list[str]):
        from .camera import Camera3D

        mode = command_parts[1].lower()
        current_cam = self.render_window.camera

        if mode == "3d":
            if isinstance(current_cam, Camera3D):
                self.output_widget.write("Already in 3D Mode.")
                return
            new_cam = Camera3D(
                position_center=current_cam.position_center,
                distance=current_cam.distance,
            )
            self.render_window.set_camera(new_cam)
            self.scene.objects = [
                obj for obj in self.scene.objects if not isinstance(obj, Grid)
            ]
            self.output_widget.write("Switched to 3D Mode")
            return

        if mode == "2d":
            if len(command_parts) < 3:
                self.output_widget.write_error(
                    "Invalid 'view 2d' command. Expected: 'view 2d <plane>'. Use 'xy', 'xz', or 'yz'."
                )
                return
            plane = command_parts[2].lower()
            self._view_2d(plane, current_cam)
            return

    def _view_2d(self, plane: str, current_cam):
        from .camera import Camera2D
        from .render_types import SnapMode

        snap_map = {"xy": SnapMode.XY, "xz": SnapMode.XZ, "yz": SnapMode.YZ}
        snap_mode = snap_map.get(plane)

        if not snap_mode:
            self.output_widget.write_error(
                f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'."
            )
            return

        if isinstance(current_cam, Camera2D) and current_cam.snap_mode == snap_mode:
            self.output_widget.write(f"Already in 2D Mode ({plane.upper()} Plane).")
            return

        new_cam = Camera2D(
            position_center=current_cam.position_center,
            distance=current_cam.distance,
            snap_mode=snap_mode,
        )
        self.render_window.set_camera(new_cam)

        self.scene.objects = [
            obj for obj in self.scene.objects if not isinstance(obj, Grid)
        ]
        self.scene.objects.append(Grid(plane=plane))

        self.output_widget.write(f"Switched to 2D Mode ({plane.upper()} Plane)")

    def _range_command(self, command_parts: list[str]):
        from .camera import Camera2D
        from .render_types import SnapMode

        if not isinstance(self.render_window.camera, Camera2D):
            self.output_widget.write_error(
                "The 'range' command is only available in 2D view modes. Use 'view 2d ...' first."
            )
            return

        if len(command_parts) == 2 and command_parts[1].lower() == "auto":
            self.render_window.clear_manual_ranges()
            self.output_widget.write("Switched to automatic ranging.")
            return

        if len(command_parts) != 4:
            self.output_widget.write_error(
                "Invalid 'range' command. Use: 'range <x|y|z> <min> <max>' or 'range auto'."
            )
            return

        axis_str = command_parts[1].lower()
        if axis_str not in ["x", "y", "z"]:
            self.output_widget.write_error(
                f"Invalid axis '{axis_str}'. Use 'x', 'y', or 'z'."
            )
            return

        try:
            min_val, max_val = float(command_parts[2]), float(command_parts[3])
        except ValueError:
            self.output_widget.write_error(
                "Invalid range values. Min and max must be numbers."
            )
            return

        if min_val >= max_val:
            self.output_widget.write_error(
                "Min range value must be less than max value."
            )
            return

        self.render_window.set_manual_range(axis_str, min_val, max_val)
        self.output_widget.write(
            f"Set manual range for {axis_str}-axis to ({min_val}, {max_val})."
        )

        # Adjust camera center and distance if both relevant ranges are now manually set
        current_snap_mode = self.render_window.camera.snap_mode
        h_axis, v_axis = None, None

        if current_snap_mode == SnapMode.XY:
            h_axis, v_axis = "x", "y"
        elif current_snap_mode == SnapMode.XZ:
            h_axis, v_axis = "x", "z"
        elif current_snap_mode == SnapMode.YZ:
            h_axis, v_axis = "z", "y"  # Z is horizontal, Y is vertical

        if (
            h_axis
            and v_axis
            and self.render_window.has_manual_range(h_axis)
            and self.render_window.has_manual_range(v_axis)
        ):
            self._adjust_camera_for_manual_ranges(h_axis, v_axis)

    def _adjust_camera_for_manual_ranges(self, h_axis: str, v_axis: str):
        manual_ranges = self.render_window.get_manual_ranges()
        h_range = manual_ranges[h_axis]
        v_range = manual_ranges[v_axis]

        # Update camera center
        axis_map = {"x": 0, "y": 1, "z": 2}
        self.render_window.camera.position_center[axis_map[h_axis]] = (
            h_range[0] + h_range[1]
        ) / 2
        self.render_window.camera.position_center[axis_map[v_axis]] = (
            v_range[0] + v_range[1]
        ) / 2

        range_width = h_range[1] - h_range[0]
        range_height = v_range[1] - v_range[0]

        if range_width <= 0 or range_height <= 0:
            self.output_widget.write_error(
                "Calculated range width or height is non-positive. Cannot adjust camera."
            )
            return

        range_aspect = range_width / range_height

        width, height = self.render_window.width(), self.render_window.height()
        if height == 0:
            self.output_widget.write_error(
                "Render widget height is zero. Cannot adjust camera aspect."
            )
            return
        window_aspect = width / height

        # Adjust camera distance to fit the range within the window, maintaining chosen aspect
        if window_aspect > range_aspect:
            # Window is wider than the desired range, fit to height
            self.render_window.camera.distance = range_height
        else:
            # Window is taller or equal aspect, fit to width
            self.render_window.camera.distance = range_width / window_aspect

        self.output_widget.write(
            f"Camera adjusted to fit manual ranges for {h_axis}/{v_axis} plane."
        )

    def _add_command(self, command_parts: list[str]):

        if len(command_parts) < 2:
            self.output_widget.write_error(
                "Invalid 'add' command format. Expected: 'add <type> ...'. Type 'help' for available commands."
            )
            return

        type_ = command_parts[1].lower()
        if type_ == "lorenz":
            self._add_lorenz()
            return

        if type_ == "nbody":
            self._add_nbody()
            return

        if type_ == "func":
            if len(command_parts) < 3:
                self.output_widget.write_error(
                    "Invalid 'add func' command. Expected: add func \"<value>\"."
                )
                return
            value_string = command_parts[2]
            self._add_func(value_string)
            return

    def _add_lorenz(self):
        from .scene import LorenzAttractor

        for obj in self.scene.objects:
            if isinstance(obj, LorenzAttractor):
                self.output_widget.write(
                    "Lorenz attractor already exists in the scene."
                )
                return
        lorenz = LorenzAttractor()
        self.scene.objects.append(lorenz)
        self.output_widget.write("Added Lorenz Attractor.")
        self.update_function_editors_callback()

    def _add_nbody(self):
        from .scene import NBody

        for obj in self.scene.objects:
            if isinstance(obj, NBody):
                self.output_widget.write(
                    "N-body simulation already exists in the scene."
                )
                return
        nbody = NBody()
        self.scene.objects.append(nbody)
        self.output_widget.write("Added N-Body Simulation.")
        self.update_function_editors_callback()

    def _add_func(self, value_string: str):
        from .scene import LinePlot, MathFunction, SurfacePlot
        from .symbolic import SymbolicFunction

        try:
            # Check if a function with this equation string already exists
            for obj in self.scene.objects:
                if isinstance(obj, MathFunction) and obj.equation_str == value_string:
                    self.output_widget.write(
                        f"Function '{value_string}' already exists in the scene."
                    )
                    return

            symbolic_func = SymbolicFunction(value_string)
            num_vars = symbolic_func.get_num_domain_vars()

            if num_vars == 1:
                new_func = LinePlot(symbolic_func)
            elif num_vars == 2:
                new_func = SurfacePlot(symbolic_func)
            else:
                self.output_widget.write_error(
                    f"Cannot plot function '{value_string}' with {num_vars} variables. Only 1 or 2 variables are supported."
                )
                return

            new_func.name = value_string  # Assign name for identification
            self.scene.objects.append(new_func)
            self.update_function_editors_callback()
            self.output_widget.write(f"Added function: {value_string}")
        except ValueError as e:
            self.output_widget.write_error(
                f"Error adding function '{value_string}': {e}"
            )

    def _remove_command(self, command_parts: list[str]):
        from .scene import MathFunction  # Import here for type checking

        if len(command_parts) < 2:
            self.output_widget.write_error(
                "Invalid 'remove' command format. Expected: 'remove <type> \"<value>\"'. Type 'help' for available commands."
            )
            return

        type_ = command_parts[1].lower()
        if type_ == "func":
            if len(command_parts) < 3:
                self.output_widget.write_error(
                    "Invalid 'remove func' command. Expected: remove func \"<value>\"."
                )
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
                self.output_widget.write(
                    f"Function '{value_string}' not found in the scene."
                )
            return
