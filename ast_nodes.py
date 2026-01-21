"""
ast_nodes.py

This file contains the class definitions for the Abstract Syntax Tree (AST) nodes.
These classes represent a mathematical expression in a tree structure.
They are pure data containers and have no knowledge of the UI.
"""

class Node:
    """The base class for all nodes in the AST."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def __repr__(self):
        return f"{self.__class__.__name__}"

class ConstantNode(Node):
    """A node representing a numerical constant (e.g., 5, 3.14)."""
    def __init__(self, value: float, parent=None):
        super().__init__(parent)
        self.value = value

    def __repr__(self):
        return f"ConstantNode(value={self.value})"

class VariableNode(Node):
    """A node representing a variable (e.g., 'x', 'y')."""
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name

    def __repr__(self):
        return f"VariableNode(name='{self.name}')"

class PlaceholderNode(Node):
    """
    A special, temporary node representing an empty, editable slot in the expression.
    This is what the user interacts with to build the tree. It is visually
    represented as an empty box or a blinking cursor.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

    def __repr__(self):
        return "PlaceholderNode()"

class BinaryOpNode(Node):
    """
    A node representing an operation with two children.
    e.g., '+', '-', '*', '/', '^'
    """
    def __init__(self, op: str, parent=None):
        super().__init__(parent)
        self.op = op
        # Children will be set after creation, e.g., [left_operand, right_operand]
        self.children = [PlaceholderNode(self), PlaceholderNode(self)]

    @property
    def left(self):
        return self.children[0] if len(self.children) > 0 else None

    @left.setter
    def left(self, node):
        if len(self.children) == 0:
            self.children.append(node)
        else:
            self.children[0] = node
        node.parent = self

    @property
    def right(self):
        return self.children[1] if len(self.children) > 1 else None

    @right.setter
    def right(self, node):
        if len(self.children) < 2:
            self.children.append(node)
        else:
            self.children[1] = node
        node.parent = self
        
    def __repr__(self):
        return f"BinaryOpNode(op='{self.op}')"

class UnaryOpNode(Node):
    """A node representing an operation with one child (e.g., negation)."""
    def __init__(self, op: str, parent=None):
        super().__init__(parent)
        self.op = op
        self.children = [PlaceholderNode(self)]

    @property
    def operand(self):
        return self.children[0]

    @operand.setter
    def operand(self, node):
        self.children[0] = node
        node.parent = self

    def __repr__(self):
        return f"UnaryOpNode(op='{self.op}')"

class FunctionNode(Node):
    """A node representing a function call, e.g., sin(x), log(x, 10)."""
    def __init__(self, name: str, arg_count: int = 1, parent=None):
        super().__init__(parent)
        self.name = name
        self.children = [PlaceholderNode(self) for _ in range(arg_count)]

    def __repr__(self):
        return f"FunctionNode(name='{self.name}')"

class EquationNode(Node):
    """The root node for a complete equation, e.g., y = x^2."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.children = [PlaceholderNode(self), PlaceholderNode(self)]

    @property
    def left(self):
        return self.children[0]

    @left.setter
    def left(self, node):
        self.children[0] = node
        node.parent = self
    
    @property
    def right(self):
        return self.children[1]

    @right.setter
    def right(self, node):
        self.children[1] = node
        node.parent = self

    def __repr__(self):
        return "EquationNode()"
