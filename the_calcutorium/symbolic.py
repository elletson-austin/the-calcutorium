from enum import Enum, auto

import numpy as np


class TokenType(Enum):
    NUMBER = auto()
    VARIABLE = auto()
    FUNCTION = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    PIPE = auto()
    EOF = auto()


FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "sec",
    "csc",
    "cot",
    "exp",
    "log",
    "ln",
    "sqrt",
    "cbrt",
    "abs",
    "floor",
    "ceil",
    "sign",
    "min",
    "max",
}

CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}


class Token:
    def __init__(self, type: TokenType, value, pos: int):
        self.type = type
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {self.value!r}, pos={self.pos})"


class TokenizeError(Exception):
    def __init__(self, message: str, pos: int):
        self.pos = pos
        super().__init__(f"{message} (at position {pos})")


def tokenize(expr: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0

    while i < len(expr):
        ch = expr[i]

        # skip whitespace
        if ch.isspace():
            i += 1
            continue

        # numbers: 3, 3.14, .5, 3e2, 3.14e-2
        if ch.isdigit() or (ch == "." and i + 1 < len(expr) and expr[i + 1].isdigit()):
            start = i
            while i < len(expr) and expr[i].isdigit():
                i += 1
            if i < len(expr) and expr[i] == ".":
                i += 1
                while i < len(expr) and expr[i].isdigit():
                    i += 1
            # scientific notation: e or E, optionally followed by + or -
            if i < len(expr) and expr[i] in ("e", "E"):
                i += 1
                if i < len(expr) and expr[i] in ("+", "-"):
                    i += 1
                if i < len(expr) and expr[i].isdigit():
                    while i < len(expr) and expr[i].isdigit():
                        i += 1
                else:
                    raise TokenizeError("Expected digit after exponent", i)
            tokens.append(Token(TokenType.NUMBER, float(expr[start:i]), start))
            continue

        # identifiers: variables, functions, constants
        if ch.isalpha() or ch == "_":
            start = i
            while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            name = expr[start:i]

            if name in FUNCTIONS:
                tokens.append(Token(TokenType.FUNCTION, name, start))
            elif name in CONSTANTS:
                tokens.append(Token(TokenType.NUMBER, CONSTANTS[name], start))
            else:
                # treat each character as a separate variable for implicit mul: "xy" -> x * y
                for j, c in enumerate(name):
                    tokens.append(Token(TokenType.VARIABLE, c, start + j))
            continue

        # single-character tokens
        simple = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.MULTIPLY,
            "/": TokenType.DIVIDE,
            "^": TokenType.POWER,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            ",": TokenType.COMMA,
            "|": TokenType.PIPE,
        }

        if ch in simple:
            # treat ** as POWER
            if ch == "*" and i + 1 < len(expr) and expr[i + 1] == "*":
                tokens.append(Token(TokenType.POWER, "**", i))
                i += 2
            else:
                tokens.append(Token(simple[ch], ch, i))
                i += 1
            continue

        raise TokenizeError(f"Unexpected character '{ch}'", i)

    tokens.append(Token(TokenType.EOF, None, len(expr)))
    return _insert_implicit_multiply(tokens)


def _insert_implicit_multiply(tokens: list[Token]) -> list[Token]:
    """Insert implicit multiplication tokens where needed.

    Cases: 2x, 2(, x(, )(, x y, )x, )2, number pi, etc.
    Rule: if left token can "end" an expression and right token can "begin" one,
    insert a * between them.

    Pipe tokens are tracked by parity: odd = opening, even = closing.
    Only closing pipes can end an expression, only opening pipes can begin one.
    """
    can_end = {TokenType.NUMBER, TokenType.VARIABLE, TokenType.RPAREN}
    can_begin = {
        TokenType.NUMBER,
        TokenType.VARIABLE,
        TokenType.FUNCTION,
        TokenType.LPAREN,
    }

    result: list[Token] = []
    pipe_count = 0

    for tok in tokens:
        if tok.type == TokenType.PIPE:
            is_opening = pipe_count % 2 == 0
            if not is_opening:
                pass
            prev_can_end = (result and result[-1].type in can_end) or (
                result and result[-1].type == TokenType.PIPE and pipe_count % 2 == 0
            )
            if is_opening and prev_can_end:
                # e.g. "2|x|" → "2 * |x|", "|x||y|" → "|x| * |y|"
                result.append(Token(TokenType.MULTIPLY, "*", tok.pos))
            if not is_opening and False:
                pass  # closing pipe handled below
            pipe_count += 1
            result.append(tok)
            continue

        left_can_end = False
        if result:
            prev = result[-1]
            if prev.type in can_end:
                left_can_end = True
            elif prev.type == TokenType.PIPE and pipe_count % 2 == 0:
                # previous was a closing pipe (pipe_count already incremented)
                left_can_end = True

        right_can_begin = tok.type in can_begin or (
            tok.type == TokenType.PIPE and pipe_count % 2 == 0
        )

        if left_can_end and right_can_begin:
            result.append(Token(TokenType.MULTIPLY, "*", tok.pos))

        result.append(tok)

    return result


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------


class NumberNode:
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"Number({self.value})"


class VariableNode:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Var({self.name})"


class BinaryOpNode:
    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class UnaryOpNode:
    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"({self.op}{self.operand})"


class FunctionCallNode:
    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


# ---------------------------------------------------------------------------
# Parser — precedence climbing / Pratt style
# ---------------------------------------------------------------------------


class ParseError(Exception):
    def __init__(self, message: str, pos: int):
        self.pos = pos
        super().__init__(f"{message} (at position {pos})")


# Precedence levels (higher = binds tighter)
_PRECEDENCE = {
    TokenType.PLUS: 1,
    TokenType.MINUS: 1,
    TokenType.MULTIPLY: 2,
    TokenType.DIVIDE: 2,
    TokenType.POWER: 3,
}


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, expected: TokenType) -> Token:
        tok = self._current()
        if tok.type != expected:
            raise ParseError(f"Expected {expected.name}, got {tok.type.name}", tok.pos)
        self.pos += 1
        return tok

    def parse(self):
        node = self._expr(0)
        if self._current().type != TokenType.EOF:
            raise ParseError(
                f"Unexpected token '{self._current().value}'", self._current().pos
            )
        return node

    def _expr(self, min_prec: int, stop_at: set[TokenType] | None = None):
        left = self._unary(stop_at)

        while (
            self._current().type in _PRECEDENCE
            and _PRECEDENCE[self._current().type] >= min_prec
            and not (stop_at and self._current().type in stop_at)
        ):
            op_tok = self._current()
            prec = _PRECEDENCE[op_tok.type]
            self.pos += 1

            # right-associative for POWER: use same prec; left-associative: use prec + 1
            next_min = prec if op_tok.type == TokenType.POWER else prec + 1
            right = self._expr(next_min, stop_at)
            left = BinaryOpNode(op_tok.value, left, right)

        return left

    def _unary(self, stop_at: set[TokenType] | None = None):
        tok = self._current()

        # unary + or -
        if tok.type in (TokenType.PLUS, TokenType.MINUS):
            self.pos += 1
            operand = self._unary(stop_at)
            if tok.type == TokenType.PLUS:
                return operand
            return UnaryOpNode("-", operand)

        return self._primary(stop_at)

    def _primary(self, stop_at: set[TokenType] | None = None):
        tok = self._current()

        # number literal
        if tok.type == TokenType.NUMBER:
            self.pos += 1
            return NumberNode(tok.value)

        # variable
        if tok.type == TokenType.VARIABLE:
            self.pos += 1
            return VariableNode(tok.value)

        # function call: name(args...)
        if tok.type == TokenType.FUNCTION:
            self.pos += 1
            self._eat(TokenType.LPAREN)
            args = [self._expr(0)]
            while self._current().type == TokenType.COMMA:
                self.pos += 1
                args.append(self._expr(0))
            self._eat(TokenType.RPAREN)
            return FunctionCallNode(tok.value, args)

        # parenthesized expression
        if tok.type == TokenType.LPAREN:
            self.pos += 1
            node = self._expr(0)
            self._eat(TokenType.RPAREN)
            return node

        # absolute value: |expr| — parse inner expr stopping at the closing pipe
        if tok.type == TokenType.PIPE:
            self.pos += 1
            node = self._expr(0, stop_at={TokenType.PIPE})
            self._eat(TokenType.PIPE)
            return FunctionCallNode("abs", [node])

        raise ParseError(f"Unexpected token '{tok.value}'", tok.pos)


def parse(expr: str):
    tokens = tokenize(expr)
    return Parser(tokens).parse()


# ---------------------------------------------------------------------------
# Compiler — AST → callable numpy function
# ---------------------------------------------------------------------------

_NUMPY_FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh,
    "sec": lambda x: 1.0 / np.cos(x),
    "csc": lambda x: 1.0 / np.sin(x),
    "cot": lambda x: np.cos(x) / np.sin(x),
    "exp": np.exp, "log": np.log, "ln": np.log,
    "sqrt": np.sqrt, "cbrt": np.cbrt,
    "abs": np.abs,
    "floor": np.floor, "ceil": np.ceil,
    "sign": np.sign,
    "min": np.minimum, "max": np.maximum,
}


def _collect_variables(node) -> set[str]:
    if isinstance(node, NumberNode):
        return set()
    if isinstance(node, VariableNode):
        return {node.name}
    if isinstance(node, UnaryOpNode):
        return _collect_variables(node.operand)
    if isinstance(node, BinaryOpNode):
        return _collect_variables(node.left) | _collect_variables(node.right)
    if isinstance(node, FunctionCallNode):
        result = set()
        for arg in node.args:
            result |= _collect_variables(arg)
        return result
    return set()


def _compile_node(node):
    """Compile an AST node into a callable that takes a dict of variable arrays."""

    if isinstance(node, NumberNode):
        val = node.value
        def _number(variables):
            return val
        return _number

    if isinstance(node, VariableNode):
        name = node.name
        def _variable(variables):
            return variables[name]
        return _variable

    if isinstance(node, UnaryOpNode):
        operand_fn = _compile_node(node.operand)
        def _negate(variables):
            return -operand_fn(variables)
        return _negate

    if isinstance(node, BinaryOpNode):
        left_fn = _compile_node(node.left)
        right_fn = _compile_node(node.right)
        op = node.op

        if op == '+':
            def _add(variables):
                return left_fn(variables) + right_fn(variables)
            return _add
        elif op == '-':
            def _sub(variables):
                return left_fn(variables) - right_fn(variables)
            return _sub
        elif op == '*':
            def _mul(variables):
                return left_fn(variables) * right_fn(variables)
            return _mul
        elif op == '/':
            def _div(variables):
                return left_fn(variables) / right_fn(variables)
            return _div
        elif op in ('^', '**'):
            def _pow(variables):
                return left_fn(variables) ** right_fn(variables)
            return _pow

    if isinstance(node, FunctionCallNode):
        np_func = _NUMPY_FUNCTIONS.get(node.name)
        if np_func is None:
            raise ValueError(f"Unknown function '{node.name}'")
        arg_fns = [_compile_node(arg) for arg in node.args]

        if len(arg_fns) == 1:
            arg_fn = arg_fns[0]
            def _call1(variables):
                return np_func(arg_fn(variables))
            return _call1
        elif len(arg_fns) == 2:
            arg_fn0, arg_fn1 = arg_fns
            def _call2(variables):
                return np_func(arg_fn0(variables), arg_fn1(variables))
            return _call2
        else:
            def _calln(variables):
                return np_func(*(fn(variables) for fn in arg_fns))
            return _calln

    raise ValueError(f"Unknown AST node type: {type(node)}")


# ---------------------------------------------------------------------------
# SymbolicFunction — public API (drop-in replacement for old sympy version)
# ---------------------------------------------------------------------------

class SymbolicFunction:
    def __init__(self, equation_str: str):
        self.equation_str = equation_str.strip()
        self.domain_vars: list[str] = []
        self.output_var: str | None = None
        self.num_domain_vars: int = 0
        self._compiled_func = None

        self._parse_and_compile()

    def _parse_and_compile(self):
        if not self.equation_str:
            raise ValueError("Empty expression")

        expr_str = self.equation_str
        self.output_var = 'y'

        if '=' in expr_str:
            parts = expr_str.split('=', 1)
            lhs = parts[0].strip()
            expr_str = parts[1].strip()
            if len(lhs) == 1 and lhs.isalpha():
                self.output_var = lhs
            else:
                raise ValueError(f"Invalid output variable '{lhs}' — must be a single letter")

        ast = parse(expr_str)
        variables = _collect_variables(ast)

        # output var should not appear as a domain var
        variables.discard(self.output_var)

        self.domain_vars = sorted(variables)
        self.num_domain_vars = len(self.domain_vars)

        if self.num_domain_vars > 2:
            raise ValueError(
                f"Too many variables ({', '.join(self.domain_vars)}). "
                f"Only 1 or 2 domain variables are supported."
            )

        self._compiled_func = _compile_node(ast)

    def evaluate(self, *args):
        if self._compiled_func is None:
            return None
        variables = {}
        for name, val in zip(self.domain_vars, args):
            variables[name] = val
        return self._compiled_func(variables)

    def get_domain_vars(self):
        return self.domain_vars

    def get_output_var(self):
        return self.output_var

    def get_num_domain_vars(self):
        return self.num_domain_vars
