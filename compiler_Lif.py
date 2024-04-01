import ast
from typing import MutableMapping, Sequence, Tuple

import utils
import x86_ast

Binding = Tuple[ast.Name, ast.expr]
Temporaries = Sequence[Binding]


class Compiler:
    def _shrink_exp(self, e: ast.expr) -> ast.expr:
        match e:
            case ast.BoolOp(ast.And(), [e1, e2]):
                e1 = self._shrink_exp(e1)
                e2 = self._shrink_exp(e2)
                return ast.IfExp(e1, e2, ast.Constant(False))
            case ast.BoolOp(ast.Or(), [e1, e2]):
                e1 = self._shrink_exp(e1)
                e2 = self._shrink_exp(e2)
                return ast.IfExp(e1, ast.Constant(True), e2)
            case _:
                return e

    def _shrink_stmt(self, s: ast.stmt) -> ast.stmt:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [exp])):
                return ast.Expr(ast.Call(ast.Name("print"), [self._shrink_exp(exp)]))
            case ast.Expr(expr):
                return ast.Expr(self._shrink_exp(expr))
            case ast.Assign([ast.Name(var)], exp):
                return ast.Assign([ast.Name(var)], self._shrink_exp(exp))
            case ast.If(exp, body, orelse):
                exp = self._shrink_exp(exp)
                body = [self._shrink_stmt(s) for s in body]
                orelse = [self._shrink_stmt(s) for s in orelse]
                return ast.If(exp, body, orelse)
            case _:
                raise Exception(f"shrink: unexpected stmt: {s}")

    def shrink(self, p: ast.Module) -> ast.Module:
        match p:
            case ast.Module(ss):
                sss = [self._shrink_stmt(s) for s in ss]
                return ast.Module(sss)
