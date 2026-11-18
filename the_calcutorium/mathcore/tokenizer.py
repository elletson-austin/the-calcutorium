"""Text → tokens.  "2xsin(y)" → [2, *, x, *, sin, (, y, )]"""

from dataclasses import dataclass
from enum import Enum, auto

from .errors import ParseError

# Names the tokenizer recognizes as functions. Every other letter is a variable.
FUNCTIONS = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "sec", "csc", "cot",
    "exp", "log", "ln", "sqrt", "cbrt", "abs",
    "floor", "ceil", "sign", "min", "max",
    "diff", "integrate",
})

CONSTANTS = {
    "pi": 3.141592653589793,
    "e": 2.718281828459045,
}

# Longest names first, so "asinh" wins over "asin" and "exp" over "e".
_KNOWN_NAMES = sorted(FUNCTIONS | CONSTANTS.keys(), key=len, reverse=True)


class T(Enum):
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


@dataclass(frozen=True)
class Token:
    type: T
    value: object
    pos: int


_SINGLE_CHAR = {
    "+": T.PLUS, "-": T.MINUS, "*": T.MULTIPLY, "/": T.DIVIDE, "^": T.POWER,
    "(": T.LPAREN, ")": T.RPAREN, ",": T.COMMA, "|": T.PIPE,
}


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(text):
        ch = text[i]

        if ch.isspace():
            i += 1
        elif ch.isdigit() or (ch == "." and i + 1 < len(text) and text[i + 1].isdigit()):
            i = _read_number(text, i, tokens)
        elif ch.isalpha():
            i = _read_word(text, i, tokens)
        elif text.startswith("**", i):
            tokens.append(Token(T.POWER, "^", i))
            i += 2
        elif ch in _SINGLE_CHAR:
            value = "^" if ch == "^" else ch
            tokens.append(Token(_SINGLE_CHAR[ch], value, i))
            i += 1
        else:
            raise ParseError(f"Unexpected character '{ch}'", i)

    tokens.append(Token(T.EOF, None, len(text)))
    return _insert_implicit_multiply(tokens)


def _read_number(text: str, i: int, tokens: list[Token]) -> int:
    """Numbers: 3, 3.14, .5, 3e2, 3.14e-2"""
    start = i
    while i < len(text) and text[i].isdigit():
        i += 1
    if i < len(text) and text[i] == ".":
        i += 1
        while i < len(text) and text[i].isdigit():
            i += 1
    # scientific notation, only if digits follow: "3e2" yes, "3e" is 3*e
    j = i
    if j < len(text) and text[j] in "eE":
        j += 1
        if j < len(text) and text[j] in "+-":
            j += 1
        if j < len(text) and text[j].isdigit():
            while j < len(text) and text[j].isdigit():
                j += 1
            i = j
    tokens.append(Token(T.NUMBER, float(text[start:i]), start))
    return i


def _read_word(text: str, i: int, tokens: list[Token]) -> int:
    """A run of letters is split into known names and single-letter variables.

    "xsin" → x, sin      "pix" → pi, x      "xy" → x, y
    """
    end = i
    while end < len(text) and text[end].isalpha():
        end += 1

    while i < end:
        name = next((n for n in _KNOWN_NAMES if text.startswith(n, i) and i + len(n) <= end), None)
        if name in FUNCTIONS:
            tokens.append(Token(T.FUNCTION, name, i))
        elif name in CONSTANTS:
            tokens.append(Token(T.NUMBER, CONSTANTS[name], i))
        else:
            name = text[i]
            tokens.append(Token(T.VARIABLE, name, i))
        i += len(name)
    return end


def _insert_implicit_multiply(tokens: list[Token]) -> list[Token]:
    """Insert `*` wherever one expression ends and the next begins: 2x, x(…), )(, |x||y|.

    Pipes are tracked by parity: an odd-numbered pipe opens |…|, an even one closes it.
    """
    can_end = {T.NUMBER, T.VARIABLE, T.RPAREN}
    can_begin = {T.NUMBER, T.VARIABLE, T.FUNCTION, T.LPAREN}

    result: list[Token] = []
    open_pipes = 0  # how many pipes seen so far
    for tok in tokens:
        is_opening_pipe = tok.type == T.PIPE and open_pipes % 2 == 0
        prev = result[-1] if result else None
        prev_ends = prev is not None and (
            prev.type in can_end or (prev.type == T.PIPE and open_pipes % 2 == 0)
        )
        this_begins = tok.type in can_begin or is_opening_pipe

        if prev_ends and this_begins:
            result.append(Token(T.MULTIPLY, "*", tok.pos))
        if tok.type == T.PIPE:
            open_pipes += 1
        result.append(tok)
    return result
