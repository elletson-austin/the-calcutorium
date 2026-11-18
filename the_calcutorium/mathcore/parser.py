"""Tokens → expression tree, by precedence climbing.

Precedence, loosest to tightest:
    + -          left-associative
    * /          left-associative (implicit multiplication too: 2x)
    unary -      so -x^2 means -(x^2), like on paper
    ^            right-associative: 2^3^2 = 2^(3^2)
"""

from .errors import ParseError
from .expr import BinOp, Call, Expr, Neg, Num, Var
from .tokenizer import T, Token, tokenize

_BINARY = {T.PLUS: 1, T.MINUS: 1, T.MULTIPLY: 2, T.DIVIDE: 2}


def parse(text: str) -> Expr:
    if not text.strip():
        raise ParseError("Empty expression", 0)
    return _Parser(tokenize(text)).parse()


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def eat(self, expected: T) -> Token:
        tok = self.current
        if tok.type != expected:
            found = "end of input" if tok.type == T.EOF else f"'{tok.value}'"
            raise ParseError(f"Expected {_describe(expected)}, found {found}", tok.pos)
        self.pos += 1
        return tok

    def parse(self) -> Expr:
        node = self.binary(0)
        if self.current.type != T.EOF:
            raise ParseError(f"Unexpected '{self.current.value}'", self.current.pos)
        return node

    def binary(self, min_prec: int) -> Expr:
        left = self.unary()
        while self.current.type in _BINARY and _BINARY[self.current.type] >= min_prec:
            op = self.current.value
            prec = _BINARY[self.current.type]
            self.pos += 1
            right = self.binary(prec + 1)
            left = BinOp(op, left, right)
        return left

    def unary(self) -> Expr:
        if self.current.type == T.PLUS:
            self.pos += 1
            return self.unary()
        if self.current.type == T.MINUS:
            self.pos += 1
            operand = self.unary()
            return Num(-operand.value) if isinstance(operand, Num) else Neg(operand)
        return self.power()

    def power(self) -> Expr:
        base = self.primary()
        if self.current.type == T.POWER:
            self.pos += 1
            exponent = self.unary()  # allows 2^-x, and 2^3^2 is right-assoc
            return BinOp("^", base, exponent)
        return base

    def primary(self) -> Expr:
        tok = self.current
        if tok.type == T.NUMBER:
            self.pos += 1
            return Num(tok.value)
        if tok.type == T.VARIABLE:
            self.pos += 1
            return Var(tok.value)
        if tok.type == T.FUNCTION:
            self.pos += 1
            self.eat(T.LPAREN)
            args = [self.binary(0)]
            while self.current.type == T.COMMA:
                self.pos += 1
                args.append(self.binary(0))
            self.eat(T.RPAREN)
            return Call(tok.value, tuple(args))
        if tok.type == T.LPAREN:
            self.pos += 1
            node = self.binary(0)
            self.eat(T.RPAREN)
            return node
        if tok.type == T.PIPE:  # |expr| is abs(expr)
            self.pos += 1
            node = self.binary(0)
            self.eat(T.PIPE)
            return Call("abs", (node,))
        found = "end of input" if tok.type == T.EOF else f"'{tok.value}'"
        raise ParseError(f"Unexpected {found}", tok.pos)


def _describe(t: T) -> str:
    return {T.LPAREN: "'('", T.RPAREN: "')'", T.PIPE: "'|'"}.get(t, t.name.lower())
