class MathError(ValueError):
    """Base class for every error the math core raises on bad user input."""


class ParseError(MathError):
    def __init__(self, message: str, pos: int):
        self.pos = pos
        super().__init__(f"{message} (at position {pos})")
