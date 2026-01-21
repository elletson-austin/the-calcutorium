from enum import Enum, auto
import numpy as np
from sympy import symbols, lambdify, sympify, SympifyError, Function
from render_types import CameraMode
from ast_nodes import (
    EquationNode, Node, ConstantNode, VariableNode, BinaryOpNode,
    PlaceholderNode, UnaryOpNode, FunctionNode
)


class Mode(Enum):
    POINTS = auto()
    LINES = auto()
    LINE_STRIP = auto()
    TRIANGLES = auto()
    LINE_LOOP = auto()


class ProgramID(Enum):
    BASIC_3D = auto()
    LORENZ_ATTRACTOR = auto()
    GRID = auto()
    SURFACE = auto()



class SceneObject:
    def __init__(self, 
                 Mode: Mode, 
                 visibility:bool = True, 
                 dynamic: bool = False,
                 ProgramID: ProgramID = ProgramID.BASIC_3D):
        self.visibility = visibility
        self.dynamic = dynamic
        self.ProgramID = ProgramID
        self.Mode = Mode

    def to_dict(self):
        return {'type': self.__class__.__name__}


class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []

    def add(self, obj: SceneObject):
        if obj in self.objects:
            raise ValueError(f"SceneObject already exists")
        self.objects.append(obj)

    def remove(self, obj: SceneObject):
        if obj not in self.objects:
            raise ValueError(f"SceneObject doesn't exist")
        self.objects.remove(obj)

    def to_dict(self):
        # Don't save the axes, it's a default part of the scene
        return [obj.to_dict() for obj in self.objects if isinstance(obj, (MathFunction, LorenzAttractor))]

    def from_dict(self, scene_data):
        # Clear existing objects except axes
        self.objects = [obj for obj in self.objects if isinstance(obj, Axes)]
        for item_data in scene_data:
            if item_data['type'] == 'MathFunction':
                new_func = MathFunction(item_data['equation'])
                new_func.name = item_data['equation']
                self.objects.append(new_func)
            elif item_data['type'] == 'LorenzAttractor':
                # You might want to save/load parameters for this in the future
                lorenz = LorenzAttractor()
                lorenz.name = "Lorenz Attractor"
                self.objects.append(lorenz)


class Axes(SceneObject):

    def __init__(self, length: float = 10.0):

        super().__init__(Mode=Mode.LINES)
        self.length = length
        # Each line is two points with RGB color

        self.vertices = np.array([
            # X axis (red)
            -length, 0, 0, 1, 0, 0,
             length, 0, 0, 1, 0, 0,
            # Y axis (green)
            0, -length, 0, 0, 1, 0,
            0,  length, 0, 0, 1, 0,
            # Z axis (blue)
            0, 0, -length, 0, 0, 1,
            0, 0,  length, 0, 0, 1,
        ], dtype=np.float32)

        self.mode = Mode.LINES
        self.dynamic = False
        self.ProgramID = ProgramID.BASIC_3D

from ast_nodes import (
    EquationNode, Node, ConstantNode, VariableNode, BinaryOpNode,
    PlaceholderNode, UnaryOpNode, FunctionNode
)
from sympy import symbols, lambdify, sympify, SympifyError, Function as SympyFunction

class MathFunction(SceneObject):
    def __init__(self, equation, points: int = 100, output_widget=None):
        super().__init__(Mode=Mode.LINE_STRIP)
        self.equation = equation
        self.points = points
        self.output_widget = output_widget
        self.equation_str = ""
        
        self.domain_vars = []
        self.output_var = None
        self._sympy_expr = None
        self._callable_func = None
        self.num_domain_vars = 0

        self.current_plane = None
        self.current_domain_range = None
        
        if isinstance(equation, str):
            self.equation_str = equation
            self._parse_string_and_lambdify()
        elif isinstance(equation, EquationNode):
            # We can create a string representation for display/debugging if needed
            self.equation_str = "AST-based function" 
            self._parse_ast_and_lambdify(equation)
        else:
            raise TypeError("Equation must be a string or an EquationNode")

        self.vertices = np.array([], dtype=np.float32)
        self.is_dirty = True

        # Initial vertex generation based on number of variables
        if self.num_domain_vars == 1:
            self.Mode = Mode.LINE_STRIP
            self.ProgramID = ProgramID.BASIC_3D
            indep_var_str = str(self.domain_vars[0])
            output_var_str = str(self.output_var)
            
            all_axes = {'x', 'y', 'z'}
            present_vars = {indep_var_str, output_var_str}
            if len(present_vars) < 2: # e.g. y=y
                const_axis_str = 'z' # arbitrary default
            else:
                const_axis_str = (all_axes - present_vars).pop()
                
            self._generate_line_vertices(domain_range=(-10, 10), indep_axis=indep_var_str, output_axis=output_var_str, const_axis=const_axis_str)
        elif self.num_domain_vars == 2:
            self.Mode = Mode.TRIANGLES
            self.ProgramID = ProgramID.SURFACE
            self._generate_surface_vertices()
        else:
            self.ProgramID = ProgramID.BASIC_3D

    def _ast_to_sympy(self, node: Node):
        if isinstance(node, ConstantNode):
            return node.value
        if isinstance(node, VariableNode):
            return symbols(node.name)
        if isinstance(node, UnaryOpNode):
            operand = self._ast_to_sympy(node.operand)
            if node.op == '-':
                return -operand
        if isinstance(node, BinaryOpNode):
            left = self._ast_to_sympy(node.left)
            right = self._ast_to_sympy(node.right)
            op_map = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a / b,
                '^': lambda a, b: a ** b,
            }
            if node.op in op_map:
                return op_map[node.op](left, right)
        if isinstance(node, FunctionNode):
            sympy_func = SympyFunction(node.name)
            args = [self._ast_to_sympy(child) for child in node.children]
            return sympy_func(*args)
        if isinstance(node, PlaceholderNode):
            from sympy import Integer
            return Integer(0) # Return sympy.Integer(0) instead of raw int 0
        raise TypeError(f"Unknown or unsupported node type for sympy conversion: {type(node)}")

    def _parse_ast_and_lambdify(self, equation_node: EquationNode):
        try:
            if not isinstance(equation_node.left, VariableNode):
                raise ValueError("Equation's left side must be a single variable (e.g., y).")
            self.output_var = symbols(equation_node.left.name)

            self._sympy_expr = self._ast_to_sympy(equation_node.right)
            self._sympy_expr = self._sympy_expr.doit()

            self.domain_vars = sorted(list(self._sympy_expr.free_symbols), key=lambda s: s.name)
            self.num_domain_vars = len(self.domain_vars)
            
            if self.num_domain_vars > 2:
                raise ValueError("Functions with more than two independent variables are not supported.")
            
            if self.num_domain_vars > 0:
                self._callable_func = lambdify(self.domain_vars, self._sympy_expr, 'numpy')

        except (ValueError, TypeError) as e:
            raise ValueError(f"Error processing equation AST: {e}") from e

    def _parse_string_and_lambdify(self):
        if not self.equation_str.strip():
            return

        try:
            output_var_str = 'y'
            expr_str = self.equation_str

            if '=' in self.equation_str:
                parts = self.equation_str.split('=', 1)
                output_var_str = parts[0].strip()
                expr_str = parts[1].strip()

            self._sympy_expr = sympify(expr_str, evaluate=False)
            self._sympy_expr = self._sympy_expr.doit()

            self.output_var = symbols(output_var_str)
            
            self.domain_vars = sorted(list(self._sympy_expr.free_symbols), key=lambda s: s.name)
            self.num_domain_vars = len(self.domain_vars)
            
            if self.num_domain_vars > 2:
                raise ValueError("Functions with more than two independent variables are not supported.")
            
            if self.num_domain_vars > 0:
                self._callable_func = lambdify(self.domain_vars, self._sympy_expr, 'numpy')

        except (SympifyError, ValueError, TypeError) as e:
            raise ValueError(f"Error processing equation: {e}") from e



    def to_dict(self):
        d = super().to_dict()
        d['equation'] = self.equation_str
        return d

    def _generate_line_vertices(self, domain_range: tuple, indep_axis: str, output_axis: str, const_axis: str = None):
        """Generates vertices for a 1-variable function."""
        vertices = []
        if self._callable_func is None or self.num_domain_vars != 1:
            self.vertices = np.array([], dtype=np.float32)
            return

        domain_values = np.linspace(domain_range[0], domain_range[1], self.points)
        
        indep_var_str = str(self.domain_vars[0])
        
        for val in domain_values:
            try:
                out_val = self._callable_func(val)
                point = {'x': 0.0, 'y': 0.0, 'z': 0.0}
                
                point[indep_axis] = val
                point[output_axis] = out_val
                
                vertices.extend([point['x'], point['y'], point['z'], 1, 1, 1]) # White color
            except Exception:
                pass 
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.is_dirty = True

    def _generate_surface_vertices(self, domain1_range=(-10, 10), domain2_range=(-10, 10)):
        """Generates a triangle mesh for a 2-variable function."""
        if self._callable_func is None or self.num_domain_vars != 2:
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        domain_var1_name = str(self.domain_vars[0])
        domain_var2_name = str(self.domain_vars[1])
        output_var_name = str(self.output_var)

        all_vars = {domain_var1_name, domain_var2_name, output_var_name}
        if len(all_vars) != 3 or not all_vars.issubset({'x', 'y', 'z'}):
            if self.output_widget:
                self.output_widget.write_error(f"Surface plot must be of the form f(a,b)=c where a,b,c are unique x,y,z variables.")
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return

        domain1_vals = np.linspace(domain1_range[0], domain1_range[1], self.points)
        domain2_vals = np.linspace(domain2_range[0], domain2_range[1], self.points)
        grid_domain1, grid_domain2 = np.meshgrid(domain1_vals, domain2_vals)

        try:
            grid_output = self._callable_func(grid_domain1, grid_domain2)
        except Exception as e:
            if self.output_widget:
                self.output_widget.write_error(f"Error evaluating surface function: {e}")
            self.vertices = np.array([], dtype=np.float32)
            self.is_dirty = True
            return
        
        grids = {
            domain_var1_name: grid_domain1,
            domain_var2_name: grid_domain2,
            output_var_name: grid_output
        }
        grid_x, grid_y, grid_z = grids['x'], grids['y'], grids['z']

        vertices = []
        color = [1.0, 0.2, 0.2] 

        for i in range(self.points - 1):
            for j in range(self.points - 1):
                p1 = np.array([grid_x[i, j], grid_y[i, j], grid_z[i, j]])
                p2 = np.array([grid_x[i, j + 1], grid_y[i, j + 1], grid_z[i, j + 1]])
                p3 = np.array([grid_x[i + 1, j], grid_y[i + 1, j], grid_z[i + 1, j]])
                p4 = np.array([grid_x[i + 1, j + 1], grid_y[i + 1, j + 1], grid_z[i + 1, j + 1]])

                # Triangle 1 (p1, p3, p2)
                v1 = p3 - p1
                v2 = p2 - p1
                n1 = np.cross(v1, v2)
                norm_n1 = np.linalg.norm(n1)
                if norm_n1 > 1e-6:
                    n1 /= norm_n1

                vertices.extend([*p1, *n1, *color])
                vertices.extend([*p3, *n1, *color])
                vertices.extend([*p2, *n1, *color])

                # Triangle 2 (p2, p3, p4)
                v1 = p3 - p2
                v2 = p4 - p2
                n2 = np.cross(v1, v2)
                norm_n2 = np.linalg.norm(n2)
                if norm_n2 > 1e-6:
                    n2 /= norm_n2

                vertices.extend([*p2, *n2, *color])
                vertices.extend([*p3, *n2, *color])
                vertices.extend([*p4, *n2, *color])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.is_dirty = True

    def regenerate(self, new_equation):
        # Clear existing sympy data and callable function
        self.domain_vars = []
        self.output_var = None
        self._sympy_expr = None
        self._callable_func = None
        self.num_domain_vars = 0

        # Set the new equation and re-parse/lambdify
        self.equation = new_equation
        if isinstance(new_equation, str):
            self.equation_str = new_equation
            self._parse_string_and_lambdify()
        elif isinstance(new_equation, EquationNode):
            self.equation_str = "AST-based function" 
            self._parse_ast_and_lambdify(new_equation)
        else:
            raise TypeError("Equation must be a string or an EquationNode")

        # Re-generate vertices based on the new equation
        if self.num_domain_vars == 1:
            self.Mode = Mode.LINE_STRIP
            self.ProgramID = ProgramID.BASIC_3D
            indep_var_str = str(self.domain_vars[0])
            output_var_str = str(self.output_var)
            
            all_axes = {'x', 'y', 'z'}
            present_vars = {indep_var_str, output_var_str}
            if len(present_vars) < 2: # e.g. y=y
                const_axis_str = 'z' # arbitrary default
            else:
                const_axis_str = (all_axes - present_vars).pop()
                
            self._generate_line_vertices(domain_range=(-10, 10), indep_axis=indep_var_str, output_axis=output_var_str, const_axis=const_axis_str)
        elif self.num_domain_vars == 2:
            self.Mode = Mode.TRIANGLES
            self.ProgramID = ProgramID.SURFACE
            self._generate_surface_vertices()
        else:
            self.ProgramID = ProgramID.BASIC_3D
        
        self.is_dirty = True


    def update_for_plane(self, plane: str, h_range: tuple, v_range: tuple):
        """Called by the renderer to update the function for the current 2D view."""
        if self.num_domain_vars != 1:
            if self.vertices.size > 0 and self.num_domain_vars > 1:
                pass # 3D surfaces are always visible
            else:
                self.vertices = np.array([], dtype=np.float32)
                self.is_dirty = True
            return

        indep_var_str = str(self.domain_vars[0])
        output_var_str = str(self.output_var)

        # Define the axis mapping for each plane: (horizontal, vertical, constant)
        plane_axis_map = {
            'xy': ('x', 'y', 'z'),
            'xz': ('x', 'z', 'y'),
            'yz': ('z', 'y', 'x')
        }
        
        h_axis, v_axis, const_axis = plane_axis_map.get(plane, ('x', 'y', 'z')) # Default to xy

        # Determine which actual axis (h_axis or v_axis) the independent variable maps to
        if indep_var_str == h_axis:
            plot_indep_axis = h_axis
            plot_output_axis = v_axis
            domain_range = h_range
        elif indep_var_str == v_axis:
            plot_indep_axis = v_axis
            plot_output_axis = h_axis
            domain_range = v_range
        else:
            # Independent variable is not on this plane, so don't draw it
            if self.vertices.size > 0:
                self.vertices = np.array([], dtype=np.float32)
                self.is_dirty = True
            return

        if self.current_plane != plane or \
           self.current_domain_range is None or \
           not np.allclose(self.current_domain_range, domain_range, atol=1e-2):
            
            self.current_plane = plane
            self.current_domain_range = domain_range
            self._generate_line_vertices(domain_range, plot_indep_axis, plot_output_axis, const_axis)
    
class Grid(SceneObject):
    def __init__(self, h_range=(-250, 250), v_range=(-250, 250), spacing=1.0, plane='xy'):
        super().__init__(Mode=Mode.LINES)
        self.h_range = h_range
        self.v_range = v_range
        self.spacing = spacing
        self.plane = plane
        self.default_h_range = h_range
        self.default_v_range = v_range
        self.default_spacing = spacing
        self.vertices = self._generate_vertices_from_ranges()
        self.ProgramID = ProgramID.GRID
        self.is_dirty = False

    def _generate_vertices_from_ranges(self):
        vertices = []
        color = [0.5, 0.5, 0.5]  # Grey color for grid lines

        start_h = self.spacing * np.floor(self.h_range[0] / self.spacing)
        end_h = self.spacing * np.ceil(self.h_range[1] / self.spacing)
        
        start_v = self.spacing * np.floor(self.v_range[0] / self.spacing)
        end_v = self.spacing * np.ceil(self.v_range[1] / self.spacing)

        axis_map = {
            'xy': (0, 1, 2),
            'xz': (0, 2, 1),
            'yz': (2, 1, 0),
        }
        h_idx, v_idx, const_idx = axis_map.get(self.plane, (0, 1, 2))
        
        # Lines along the horizontal axis
        for i in np.arange(start_v, end_v + self.spacing, self.spacing):
            p1 = [0, 0, 0]
            p2 = [0, 0, 0]
            p1[h_idx] = self.h_range[0]
            p2[h_idx] = self.h_range[1]
            p1[v_idx] = i
            p2[v_idx] = i
            vertices.extend(p1 + color)
            vertices.extend(p2 + color)

        # Lines along the vertical axis
        for i in np.arange(start_h, end_h + self.spacing, self.spacing):
            p1 = [0, 0, 0]
            p2 = [0, 0, 0]
            p1[v_idx] = self.v_range[0]
            p2[v_idx] = self.v_range[1]
            p1[h_idx] = i
            p2[h_idx] = i
            vertices.extend(p1 + color)
            vertices.extend(p2 + color)
            
        return np.array(vertices, dtype=np.float32)

    def set_ranges(self, h_range, v_range, plane='xy'):
        view_size = min(h_range[1] - h_range[0], v_range[1] - v_range[0])
        
        if view_size <= 0:
            return
            
        target_spacing = view_size / 10.0
        
        power_of_10 = 10**np.floor(np.log10(target_spacing))
        rescaled_spacing = target_spacing / power_of_10

        if rescaled_spacing < 1.5:
            new_spacing = 1 * power_of_10
        elif rescaled_spacing < 3.5:
            new_spacing = 2 * power_of_10
        elif rescaled_spacing < 7.5:
            new_spacing = 5 * power_of_10
        else:
            new_spacing = 10 * power_of_10
            
        if not np.isclose(self.spacing, new_spacing) or \
           not np.allclose(self.h_range, h_range) or \
           not np.allclose(self.v_range, v_range) or \
           self.plane != plane:
            
            self.spacing = new_spacing
            self.h_range = h_range
            self.v_range = v_range
            self.plane = plane
            
            self.vertices = self._generate_vertices_from_ranges()
            self.is_dirty = True

    def set_to_default(self):
        if not (np.allclose(self.h_range, self.default_h_range) and \
                np.allclose(self.v_range, self.default_v_range) and \
                np.isclose(self.spacing, self.default_spacing) and \
                self.plane != 'xy'):
            self.h_range = self.default_h_range
            self.v_range = self.default_v_range
            self.spacing = self.default_spacing
            self.plane = 'xy'
            self.vertices = self._generate_vertices_from_ranges()
            self.is_dirty = True

class LorenzAttractor(SceneObject):
    def __init__(self, num_points: int = 100_000, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0, dt: float = 0.001, steps:int = 5):
        super().__init__(Mode=Mode.POINTS, ProgramID=ProgramID.LORENZ_ATTRACTOR, dynamic=True)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.steps = steps
        self.num_points = num_points
        self.vertices = self.create_initial_points(num_points=self.num_points)

    def to_dict(self):
        return super().to_dict()

    def create_initial_points(self,num_points: int) -> np.ndarray:
        # Note: We are not including color data here, as the fragment shader will assign a color.
        # The vertex format is just position (x, y, z, w).
        initial_points = np.random.randn(num_points, 4).astype(np.float32)
        initial_points[:, :3] *= 2.0
        initial_points[:, :3] += [1.0, 1.0, 1.0]
        initial_points[:, 3] = 1.0
        return initial_points