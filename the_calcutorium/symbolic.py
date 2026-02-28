from sympy import (
    symbols, lambdify, sympify, SympifyError,
    sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, log, sqrt, cbrt,
    sec, csc, cot, asec, acsc, acot, atan2,
    asinh, acosh, atanh, asech, acsch, acoth,
    Abs, factorial, factorial2, gamma, Min, Max, sign,
    floor, ceiling, frac,
    binomial, fibonacci, lucas,
    gcd, lcm, isprime, prime
)

class SymbolicFunction:
    def __init__(self, equation_str: str):
        self.equation_str = equation_str
        self.domain_vars = []
        self.output_var = None
        self._sympy_expr = None
        self._callable_func = None
        self.num_domain_vars = 0

        self._parse_and_lambdify()

    def _parse_and_lambdify(self):
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

    def evaluate(self, *args):
        if self._callable_func:
            return self._callable_func(*args)
        return None

    def get_domain_vars(self):
        return self.domain_vars

    def get_output_var(self):
        return self.output_var

    def get_num_domain_vars(self):
        return self.num_domain_vars
