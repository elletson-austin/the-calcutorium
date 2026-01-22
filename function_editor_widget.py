"""
function_editor_widget.py

This file will contain the implementation for the visual Abstract Syntax Tree (AST) editor.
The architecture is composed of three main parts:
1.  The Model: A set of classes that define the AST itself.
2.  The View: A custom widget that knows how to render the AST graphically.
3.  The Controller: Logic that handles user input (from a virtual keypad or keyboard)
    to modify the AST.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QKeyEvent, QPainter, QColor, QFont, QMouseEvent, QFontMetrics
from PySide6.QtCore import Signal, Qt, QRectF, QPointF

from ast_nodes import (
    Node, ConstantNode, VariableNode, BinaryOpNode, PlaceholderNode,
    UnaryOpNode, FunctionNode, EquationNode
)
from scene import MathFunction



# --- 2. The View: AST Renderer Widget ---
class ExpressionView(QWidget):
    """
    A widget that graphically renders a single Abstract Syntax Tree (AST).
    This class is responsible for all custom painting and layout logic.
    """
    ast_changed = Signal(object) # Emit the view itself so editor knows which one changed

    def __init__(self, ast_root: Node, parent=None):
        super().__init__(parent)
        self.ast_root = ast_root
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.cursor_node = ast_root
        
        self.font = QFont("Arial", 14)
        self.op_font = QFont("Arial", 14)
        self.sup_font = QFont("Arial", 10)
        self.padding = 5 # General padding

        # This will store the calculated layout rectangles for each node
        self.layout_rects = {}
        # This will store the final screen rectangles for hit testing
        self.node_render_rects = {}

        self.setMinimumHeight(50)
    
    def _ast_to_string(self, node: Node) -> str:
        """Recursively traverses the AST to build a SymPy-compatible string."""
        if isinstance(node, ConstantNode):
            return str(node.value)
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, PlaceholderNode):
            # Use text if available, otherwise treat as 0 for continuity
            return node.text or "0"
        if isinstance(node, UnaryOpNode):
            # Parenthesize to ensure correct precedence
            return f"({node.op}{self._ast_to_string(node.operand)})"
        if isinstance(node, BinaryOpNode):
            # Use Python's power operator ** instead of ^
            op_str = "**" if node.op == '^' else node.op
            # Parenthesize to ensure correct precedence
            return f"({self._ast_to_string(node.left)} {op_str} {self._ast_to_string(node.right)})"
        if isinstance(node, FunctionNode):
            args = ", ".join([self._ast_to_string(child) for child in node.children])
            return f"{node.name}({args})"
        if isinstance(node, EquationNode):
            left_str = self._ast_to_string(node.left)
            right_str = self._ast_to_string(node.right)
            # Default to 'y' if left side is empty/invalid during editing
            if not left_str or left_str == "0":
                left_str = "y"
            return f"{left_str} = {right_str}"
        return ""

    def _commit_placeholder_text(self, node):
        """
        If the given node is a PlaceholderNode with text, convert it to a
        ConstantNode or VariableNode and return the new node. Otherwise,
        return the original node.
        """
        if isinstance(node, PlaceholderNode) and node.text:
            text = node.text
            try:
                # Try to convert to a number first
                value = float(text)
                new_node = ConstantNode(value, parent=node.parent)
            except ValueError:
                # Otherwise, it's a variable
                new_node = VariableNode(text, parent=node.parent)
            
            self._replace_node(node, new_node)
            self.ast_changed.emit(self) # Notify the scene of the change
            return new_node
        return node

    def _get_in_order_nodes(self, node: Node, node_list: list):
        """Performs an in-order traversal of the AST to get a flat list of nodes."""
        if isinstance(node, (BinaryOpNode, EquationNode)):
            self._get_in_order_nodes(node.left, node_list)
            node_list.append(node)
            self._get_in_order_nodes(node.right, node_list)
        elif isinstance(node, UnaryOpNode):
            node_list.append(node)
            self._get_in_order_nodes(node.operand, node_list)
        elif isinstance(node, FunctionNode):
            node_list.append(node)
            for child in node.children:
                self._get_in_order_nodes(child, node_list)
        else: # For leaf nodes like ConstantNode, VariableNode, PlaceholderNode
            node_list.append(node)

    def _replace_node(self, old_node: Node, new_node: Node):
        """Replaces old_node with new_node in the AST."""
        parent = old_node.parent
        if parent is None: # old_node is the root
            self.ast_root = new_node
        else:
            for i, child in enumerate(parent.children):
                if child == old_node:
                    parent.children[i] = new_node
                    break
        new_node.parent = parent
        self.cursor_node = new_node # Move cursor to the newly created node

    def keyPressEvent(self, event: QKeyEvent):
        """Handles user input for navigation and editing."""
        
        # --- Pre-action: Commit any pending text in a placeholder ---
        # This is crucial for turning typed text like "x" into a VariableNode
        # before an action like moving the cursor or inserting an operator.
        if event.key() in [Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down] \
           or event.text() in ['+', '-', '*', '/', '^']:
            self.cursor_node = self._commit_placeholder_text(self.cursor_node)

        nodes_in_order = []
        self._get_in_order_nodes(self.ast_root, nodes_in_order)
        
        try:
            current_index = nodes_in_order.index(self.cursor_node)
        except ValueError:
            # If cursor node isn't in the list (e.g., after a commit), find its new position
            try:
                # This is a bit of a guess, assuming the committed node is what we want
                current_index = nodes_in_order.index(self.cursor_node)
            except ValueError:
                current_index = 0

        # --- Handle navigation and actions ---

        if event.key() == Qt.Key.Key_Left:
            new_index = max(0, current_index - 1)
            self.cursor_node = nodes_in_order[new_index]
            self.update()
            event.accept()

        elif event.key() == Qt.Key.Key_Right:
            new_index = min(len(nodes_in_order) - 1, current_index + 1)
            self.cursor_node = nodes_in_order[new_index]
            self.update()
            event.accept()

        elif event.key() == Qt.Key.Key_Backspace:
            if isinstance(self.cursor_node, PlaceholderNode) and self.cursor_node.text:
                # If there's text, just delete the last character
                self.cursor_node.text = self.cursor_node.text[:-1]
            elif not isinstance(self.cursor_node, EquationNode):
                # If no text or not a placeholder, replace the node
                new_node = PlaceholderNode(parent=self.cursor_node.parent)
                self._replace_node(self.cursor_node, new_node)
            
            self.ast_changed.emit(self)
            self.update()
            event.accept()

        elif event.text() in ['+', '-', '*', '/', '^']:
            op = event.text()
            if isinstance(self.cursor_node, (ConstantNode, VariableNode, BinaryOpNode, FunctionNode)):
                node_to_wrap = self.cursor_node
                original_parent = node_to_wrap.parent

                if original_parent is None or (isinstance(original_parent, EquationNode) and original_parent.left == node_to_wrap):
                    super().keyPressEvent(event)
                    return
                
                try:
                    idx = original_parent.children.index(node_to_wrap)
                except ValueError:
                    super().keyPressEvent(event)
                    return

                op_node = BinaryOpNode(op)
                op_node.parent = original_parent
                op_node.left = node_to_wrap
                op_node.right = PlaceholderNode(op_node)
                node_to_wrap.parent = op_node
                original_parent.children[idx] = op_node
                self.cursor_node = op_node.right
                
                self.ast_changed.emit(self)
                self.update()
                event.accept()

        elif event.text() == '(':
            if isinstance(self.cursor_node, PlaceholderNode) and self.cursor_node.text:
                func_name = self.cursor_node.text
                new_node = FunctionNode(name=func_name, parent=self.cursor_node.parent)
                self._replace_node(self.cursor_node, new_node)
                self.cursor_node = new_node.children[0]
                self.ast_changed.emit(self)
                self.update()
                event.accept()
            else:
                # TODO: Handle parentheses wrapping a selection
                super().keyPressEvent(event)

        elif isinstance(self.cursor_node, PlaceholderNode):
            text = event.text()
            if text.isalnum():
                self.cursor_node.text += text
                self.update()
                event.accept()
            elif Qt.Key_0 <= event.key() <= Qt.Key_9:
                 # This case might be redundant now with isalnum, but kept for safety
                self.cursor_node.text += text
                self.update()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)


    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse clicks to set the cursor node."""
        click_pos = event.position()
        
        self.cursor_node = self._commit_placeholder_text(self.cursor_node)

        clicked_nodes = []
        for node, rect in self.node_render_rects.items():
            if rect.contains(click_pos):
                clicked_nodes.append(node)
        
        if clicked_nodes:
            clicked_nodes.sort(key=lambda n: self.node_render_rects[n].width() * self.node_render_rects[n].height())
            self.cursor_node = clicked_nodes[0]
            self.setFocus()
            self.update()
            event.accept()
        else:
            super().mousePressEvent(event)

    def paintEvent(self, event):
        """The main entry point for drawing the widget."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        
        painter.setPen(QColor("red"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(Qt.GlobalColor.black)

        self.layout_rects.clear()
        self.node_render_rects.clear()
        
        total_rect = self._calculate_rect(self.ast_root)
        
        start_y = (self.height() - total_rect.height()) / 2
        
        self._draw_node(painter, self.ast_root, QPointF(self.padding, start_y))

    def _calculate_rect(self, node: Node) -> QRectF:
        """
        Recursively calculates the bounding rectangle for a node and all its children.
        Returns the bounding QRectF for the given node.
        """
        if node in self.layout_rects:
            return self.layout_rects[node]

        metrics = QFontMetrics(self.font)
        rect = QRectF()

        if isinstance(node, (ConstantNode, VariableNode)):
            text = str(node.value) if isinstance(node, ConstantNode) else node.name
            rect = QRectF(0, 0, metrics.horizontalAdvance(text), metrics.height())
        elif isinstance(node, PlaceholderNode):
            if node.text:
                rect = QRectF(0, 0, metrics.horizontalAdvance(node.text), metrics.height())
            else:
                rect = QRectF(0, 0, 15, metrics.height())
        elif isinstance(node, (BinaryOpNode, EquationNode)):
            if isinstance(node, BinaryOpNode) and node.op == '^':
                left_rect = self._calculate_rect(node.left)
                
                original_font = self.font
                self.font = self.sup_font
                right_rect = self._calculate_rect(node.right)
                self.font = original_font
                
                total_width = left_rect.width() + right_rect.width() + self.padding / 2
                total_height = left_rect.height() + right_rect.height() * 0.5
                rect = QRectF(0, 0, total_width, total_height)
            else:
                left_rect = self._calculate_rect(node.left)
                right_rect = self._calculate_rect(node.right)
                
                op_text = f" {node.op} " if isinstance(node, BinaryOpNode) else " = "
                op_width = metrics.horizontalAdvance(op_text)
                
                if isinstance(node, BinaryOpNode) and node.op == '/':
                    total_width = max(left_rect.width(), right_rect.width())
                    total_height = left_rect.height() + self.padding + metrics.height() + self.padding + right_rect.height()
                    rect = QRectF(0, 0, total_width, total_height)
                else:
                    total_width = left_rect.width() + op_width + right_rect.width()
                    max_height = max(left_rect.height(), right_rect.height())
                    rect = QRectF(0, 0, total_width, max_height)
        elif isinstance(node, UnaryOpNode):
            operand_rect = self._calculate_rect(node.operand)
            op_text = node.op
            op_width = metrics.horizontalAdvance(op_text)
            rect = QRectF(0, 0, op_width + operand_rect.width(), operand_rect.height())
        elif isinstance(node, FunctionNode):
            func_name_width = metrics.horizontalAdvance(node.name + "(")
            closing_paren_width = metrics.horizontalAdvance(")")
            
            args_width = 0
            max_arg_height = 0
            for i, child in enumerate(node.children):
                child_rect = self._calculate_rect(child)
                args_width += child_rect.width()
                max_arg_height = max(max_arg_height, child_rect.height())
                if i < len(node.children) - 1:
                    args_width += metrics.horizontalAdvance(", ")
            total_width = func_name_width + args_width + closing_paren_width
            total_height = max(metrics.height(), max_arg_height)
            rect = QRectF(0, 0, total_width, total_height)
            
        self.layout_rects[node] = rect
        return rect

    def _draw_node(self, painter: QPainter, node: Node, position: QPointF):
        """
        Recursively draws a node and its children.
        `position` is the top-left corner where drawing should start.
        """
        node_rect = self.layout_rects.get(node, QRectF())
        
        final_rect = QRectF(position, node_rect.size())
        self.node_render_rects[node] = final_rect

        if node == self.cursor_node:
            painter.setBrush(QColor(0, 100, 255, 60))
            painter.setPen(Qt.PenStyle.NoPen)
            highlight_rect = final_rect.adjusted(-2, -2, 2, 2)
            painter.drawRoundedRect(highlight_rect, 3, 3)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(Qt.GlobalColor.black)

        text_v_center = position.y() + node_rect.height() / 2
        
        # Use the correct font for the current painter state
        metrics = painter.fontMetrics()

        if isinstance(node, ConstantNode):
            painter.setFont(self.font)
            painter.drawText(QPointF(position.x(), text_v_center + metrics.ascent() / 2), str(node.value))
        elif isinstance(node, VariableNode):
            painter.setFont(self.font)
            painter.drawText(QPointF(position.x(), text_v_center + metrics.ascent() / 2), node.name)
        elif isinstance(node, PlaceholderNode):
            if node.text:
                painter.setFont(self.font)
                painter.drawText(QPointF(position.x(), text_v_center + metrics.ascent() / 2), node.text)
            else:
                painter.setBrush(QColor(240, 240, 240))
                painter.setPen(Qt.PenStyle.DotLine)
                painter.drawRect(final_rect)
        elif isinstance(node, (BinaryOpNode, EquationNode)):
            left_rect = self.layout_rects.get(node.left, QRectF())
            right_rect = self.layout_rects.get(node.right, QRectF())
            
            if isinstance(node, BinaryOpNode) and node.op == '^':
                left_child_y = position.y() + right_rect.height() * 0.5
                self._draw_node(painter, node.left, QPointF(position.x(), left_child_y))
                
                original_font = painter.font()
                painter.setFont(self.sup_font)
                right_child_y = position.y()
                self._draw_node(painter, node.right, QPointF(position.x() + left_rect.width() + self.padding / 2, right_child_y))
                painter.setFont(original_font)
            else:
                op_text = f" {node.op} " if isinstance(node, BinaryOpNode) else " = "
                
                if isinstance(node, BinaryOpNode) and node.op == '/':
                    left_start_x = position.x() + (node_rect.width() - left_rect.width()) / 2
                    self._draw_node(painter, node.left, QPointF(left_start_x, position.y()))
                    
                    bar_y = position.y() + left_rect.height() + self.padding / 2
                    painter.drawLine(QPointF(position.x(), bar_y), QPointF(position.x() + node_rect.width(), bar_y))

                    right_start_x = position.x() + (node_rect.width() - right_rect.width()) / 2
                    self._draw_node(painter, node.right, QPointF(right_start_x, bar_y + self.padding / 2 + metrics.height()))
                else:
                    left_child_y = position.y() + (node_rect.height() - left_rect.height()) / 2
                    self._draw_node(painter, node.left, QPointF(position.x(), left_child_y))
                    
                    current_x = position.x() + left_rect.width()
                    painter.setFont(self.op_font)
                    painter.drawText(QPointF(current_x, text_v_center + metrics.ascent() / 2), op_text)

                    current_x += metrics.horizontalAdvance(op_text)
                    right_child_y = position.y() + (node_rect.height() - right_rect.height()) / 2
                    self._draw_node(painter, node.right, QPointF(current_x, right_child_y))
        elif isinstance(node, UnaryOpNode):
            op_text = node.op
            op_width = metrics.horizontalAdvance(op_text)
            
            painter.setFont(self.op_font)
            painter.drawText(QPointF(position.x(), text_v_center + metrics.ascent() / 2), op_text)
            
            self._draw_node(painter, node.operand, QPointF(position.x() + op_width, position.y()))
        elif isinstance(node, FunctionNode):
            current_x = position.x()
            
            painter.drawText(QPointF(current_x, text_v_center + metrics.ascent() / 2), node.name + "(")
            current_x += metrics.horizontalAdvance(node.name + "(")
            
            for i, child in enumerate(node.children):
                child_rect = self.layout_rects.get(child, QRectF())
                child_y = position.y() + (node_rect.height() - child_rect.height()) / 2
                self._draw_node(painter, child, QPointF(current_x, child_y))
                current_x += child_rect.width()
                if i < len(node.children) - 1:
                    painter.drawText(QPointF(current_x, text_v_center + metrics.ascent() / 2), ", ")
                    current_x += metrics.horizontalAdvance(", ")
            
            painter.drawText(QPointF(current_x, text_v_center + metrics.ascent() / 2), ")")


class ExpressionEditor(QWidget):
    """
    The main container widget for the entire visual expression editor.
    It manages a list of expressions and orchestrates user interaction.
    """
    def __init__(self, parent=None, scene=None):
        super().__init__(parent)
        self.scene = scene
        self.view_to_function_map = {}
        
        main_layout = QVBoxLayout(self)
        self.expressions_list_widget = QWidget()
        self.expressions_layout = QVBoxLayout(self.expressions_list_widget)
        main_layout.addWidget(self.expressions_list_widget)

        self.expression_views = []
        self.active_view = None

        self.add_new_expression()

    def add_new_expression(self):
        """Creates a new, empty expression and adds it to the editor."""
        equation = EquationNode()
        equation.left = VariableNode('y')
        equation.right = PlaceholderNode(parent=equation) # y = [placeholder]

        # Create the visual widget for the expression
        view = ExpressionView(equation)
        self.expressions_layout.addWidget(view)
        self.expression_views.append(view)
        self.set_active_view(view)

        # Connect the ast_changed signal
        view.ast_changed.connect(self._on_ast_changed)

        # If the scene is connected, create and add the corresponding MathFunction
        if self.scene:
            try:
                # On creation, pass the initial string representation
                equation_str = view._ast_to_string(equation)
                new_func = MathFunction(equation_str=equation_str)
                self.scene.objects.append(new_func)
                self.view_to_function_map[view] = new_func
            except Exception as e:
                # TODO: Handle this error in the UI, e.g., in the output widget
                print(f"Error creating function from AST string: {e}")

    def set_active_view(self, view: ExpressionView):
        """Sets the currently active expression view for keyboard input."""
        self.active_view = view
        view.setFocus()
        # TODO: Add visual indication for the active view (e.g., a border).
    
    def _on_ast_changed(self, changed_view: ExpressionView):
        """Slot to handle changes in the AST of an ExpressionView."""
        if self.scene and changed_view in self.view_to_function_map:
            math_function = self.view_to_function_map[changed_view]
            try:
                # Convert the entire equation AST to a string and regenerate
                equation_str = changed_view._ast_to_string(changed_view.ast_root)
                math_function.regenerate(equation_str)
            except Exception as e:
                # TODO: Display this error in the output widget for the user
                print(f"Error regenerating from AST string: {e}")