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
        """Handles user input for navigation and (soon) editing."""
        nodes_in_order = []
        self._get_in_order_nodes(self.ast_root, nodes_in_order)
        
        try:
            current_index = nodes_in_order.index(self.cursor_node)
        except ValueError:
            current_index = 0

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
        elif event.key() == Qt.Key.Key_Backspace and not isinstance(self.cursor_node, EquationNode):
            new_node = PlaceholderNode(parent=self.cursor_node.parent)
            self._replace_node(self.cursor_node, new_node)
            self.ast_changed.emit(self)
            self.update()
            event.accept()
        elif event.text() in ['+', '-', '*', '/', '^']:
            # Handle operator insertion
            op = event.text()
            # We can insert an operator if the cursor is on a "wrappable" node
            if isinstance(self.cursor_node, (ConstantNode, VariableNode, BinaryOpNode, FunctionNode)):
                node_to_wrap = self.cursor_node
                original_parent = node_to_wrap.parent

                # Abort if we can't find a parent to modify
                if original_parent is None:
                    super().keyPressEvent(event)
                    return

                # Don't wrap the 'y' variable in 'y = ...'
                if isinstance(original_parent, EquationNode) and original_parent.left == node_to_wrap:
                    super().keyPressEvent(event)
                    return

                # Find the index of the node we're wrapping
                try:
                    idx = original_parent.children.index(node_to_wrap)
                except ValueError:
                    super().keyPressEvent(event)
                    return

                # Perform the tree surgery
                op_node = BinaryOpNode(op)
                op_node.parent = original_parent
                op_node.left = node_to_wrap
                op_node.right = PlaceholderNode(op_node) # Set parent on creation
                node_to_wrap.parent = op_node

                # Replace the old node with the new operator node in the parent
                original_parent.children[idx] = op_node

                # Move cursor to the new placeholder
                self.cursor_node = op_node.right
                
                self.ast_changed.emit(self)
                self.update()
                event.accept()

        elif isinstance(self.cursor_node, PlaceholderNode):
            if Qt.Key_0 <= event.key() <= Qt.Key_9:
                value = int(event.text())
                new_node = ConstantNode(value, parent=self.cursor_node.parent)
                self._replace_node(self.cursor_node, new_node)
                self.ast_changed.emit(self)
                self.update()
                event.accept()
            elif event.text().isalpha() and len(event.text()) == 1:
                name = event.text().lower()
                new_node = VariableNode(name, parent=self.cursor_node.parent)
                self._replace_node(self.cursor_node, new_node)
                self.ast_changed.emit(self)
                self.update()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse clicks to set the cursor node."""
        click_pos = event.position()
        
        # Find all nodes that contain the click position
        clicked_nodes = []
        for node, rect in self.node_render_rects.items():
            if rect.contains(click_pos):
                clicked_nodes.append(node)
        
        # Of the nodes that were clicked, choose the one with the smallest area
        if clicked_nodes:
            # Sort by area (width * height)
            clicked_nodes.sort(key=lambda n: self.node_render_rects[n].width() * self.node_render_rects[n].height())
            self.cursor_node = clicked_nodes[0]
            self.setFocus() # Ensure widget has focus for keyboard input
            self.update() # Redraw to show new cursor
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

        # Clear old layout data before recalculating
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

        metrics = self.fontMetrics()
        rect = QRectF()

        if isinstance(node, (ConstantNode, VariableNode)):
            text = str(node.value) if isinstance(node, ConstantNode) else node.name
            rect = QRectF(0, 0, metrics.horizontalAdvance(text), metrics.height())
        elif isinstance(node, PlaceholderNode):
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
                op_width = self.fontMetrics().horizontalAdvance(op_text)
                
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
            op_width = self.fontMetrics().horizontalAdvance(op_text)
            rect = QRectF(0, 0, op_width + operand_rect.width(), operand_rect.height())
        elif isinstance(node, FunctionNode):
            func_name_width = self.fontMetrics().horizontalAdvance(node.name + "(")
            closing_paren_width = self.fontMetrics().horizontalAdvance(")")
            
            args_width = 0
            max_arg_height = 0
            for i, child in enumerate(node.children):
                child_rect = self._calculate_rect(child)
                args_width += child_rect.width()
                max_arg_height = max(max_arg_height, child_rect.height())
                if i < len(node.children) - 1:
                    args_width += self.fontMetrics().horizontalAdvance(", ")
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
        
        # Store the final screen rectangle for hit testing
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

        if isinstance(node, ConstantNode):
            painter.setFont(self.font)
            painter.drawText(QPointF(position.x(), text_v_center + self.fontMetrics().ascent() / 2), str(node.value))
        elif isinstance(node, VariableNode):
            painter.setFont(self.font)
            painter.drawText(QPointF(position.x(), text_v_center + self.fontMetrics().ascent() / 2), node.name)
        elif isinstance(node, PlaceholderNode):
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
                op_width = self.fontMetrics().horizontalAdvance(op_text)
                
                if isinstance(node, BinaryOpNode) and node.op == '/':
                    left_start_x = position.x() + (node_rect.width() - left_rect.width()) / 2
                    self._draw_node(painter, node.left, QPointF(left_start_x, position.y()))
                    
                    bar_y = position.y() + left_rect.height() + self.padding / 2
                    painter.drawLine(QPointF(position.x(), bar_y), QPointF(position.x() + node_rect.width(), bar_y))

                    right_start_x = position.x() + (node_rect.width() - right_rect.width()) / 2
                    self._draw_node(painter, node.right, QPointF(right_start_x, bar_y + self.padding / 2 + self.fontMetrics().height()))
                else:
                    left_child_y = position.y() + (node_rect.height() - left_rect.height()) / 2
                    self._draw_node(painter, node.left, QPointF(position.x(), left_child_y))
                    
                    current_x = position.x() + left_rect.width()
                    painter.setFont(self.op_font)
                    painter.drawText(QPointF(current_x, text_v_center + self.fontMetrics().ascent() / 2), op_text)

                    current_x += op_width
                    right_child_y = position.y() + (node_rect.height() - right_rect.height()) / 2
                    self._draw_node(painter, node.right, QPointF(current_x, right_child_y))
        elif isinstance(node, UnaryOpNode):
            op_text = node.op
            op_width = self.fontMetrics().horizontalAdvance(op_text)
            
            painter.setFont(self.op_font)
            painter.drawText(QPointF(position.x(), text_v_center + self.fontMetrics().ascent() / 2), op_text)
            
            self._draw_node(painter, node.operand, QPointF(position.x() + op_width, position.y()))
        elif isinstance(node, FunctionNode):
            current_x = position.x()
            
            painter.drawText(QPointF(current_x, text_v_center + self.fontMetrics().ascent() / 2), node.name + "(")
            current_x += self.fontMetrics().horizontalAdvance(node.name + "(")
            
            for i, child in enumerate(node.children):
                child_rect = self.layout_rects.get(child, QRectF())
                child_y = position.y() + (node_rect.height() - child_rect.height()) / 2
                self._draw_node(painter, child, QPointF(current_x, child_y))
                current_x += child_rect.width()
                if i < len(node.children) - 1:
                    painter.drawText(QPointF(current_x, text_v_center + self.fontMetrics().ascent() / 2), ", ")
                    current_x += self.fontMetrics().horizontalAdvance(", ")
            
            painter.drawText(QPointF(current_x, text_v_center + self.fontMetrics().ascent() / 2), ")")


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
                # The initial function is basically empty, so it might not plot anything
                # or might evaluate to y=0 depending on placeholder handling. This is expected.
                new_func = MathFunction(equation=equation)
                self.scene.objects.append(new_func)
                self.view_to_function_map[view] = new_func
            except Exception as e:
                # TODO: Handle this error in the UI, e.g., in the output widget
                print(f"Error creating function from AST: {e}")

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
                math_function.regenerate(changed_view.ast_root)
            except Exception as e:
                # TODO: Display this error in the output widget for the user
                print(f"Error regenerating MathFunction from AST: {e}")