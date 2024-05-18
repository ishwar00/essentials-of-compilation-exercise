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

    def rco_exp(self, e: ast.expr, need_atomic: bool) -> Tuple[ast.expr, Temporaries]:
        match e:
            case ast.Constant(v):
                return ast.Constant(v), []
            case ast.Name(var):
                return ast.Name(var), []
            case ast.Call(ast.Name("input_int"), []):
                if need_atomic:
                    tmp = utils.generate_name("tmp")
                    return (
                        ast.Name(tmp),
                        [(ast.Name(tmp), ast.Call(ast.Name("input_int"), []))],
                    )
                return (ast.Call(ast.Name("input_int"), []), [])
            case ast.UnaryOp(ast.USub() | ast.Not() as op, condition):
                (atm, temps) = self.rco_exp(condition, True)
                if need_atomic:
                    tmp = utils.generate_name("tmp")
                    return (
                        ast.Name(tmp),
                        [*temps, (ast.Name(tmp), ast.UnaryOp(ast.USub(), atm))],
                    )
                return (ast.UnaryOp(op, atm), temps)
            case ast.BinOp(exp1, op, exp2):
                (atm1, temps1) = self.rco_exp(exp1, True)
                (atm2, temps2) = self.rco_exp(exp2, True)
                if need_atomic:
                    tmp = utils.generate_name("tmp")
                    return (
                        ast.Name(tmp),
                        [
                            *temps1,
                            *temps2,
                            (ast.Name(tmp), ast.BinOp(atm1, op, atm2)),
                        ],
                    )

                return (ast.BinOp(atm1, op, atm2), [*temps1, *temps2])
            case ast.Compare(exp1, [op], [exp2]):
                (atm1, temps1) = self.rco_exp(exp1, True)
                (atm2, temps2) = self.rco_exp(exp2, True)
                if need_atomic:
                    tmp = utils.generate_name("tmp")
                    return (
                        ast.Name(tmp),
                        [
                            *temps1,
                            *temps2,
                            (ast.Name(tmp), ast.Compare(atm1, [op], [atm2])),
                        ],
                    )

                return (ast.Compare(atm1, [op], [atm2]), [*temps1, *temps2])
            case ast.IfExp(condition, body, orelse):
                (simplified_cond, cond_temps) = self.rco_exp(condition, False)

                (body_expr, body_temps) = self.rco_exp(body, True)
                body_expr = utils.Begin(
                    [ast.Assign([var], value) for (var, value) in body_temps],
                    body_expr,
                )

                (orelse_expr, orelse_temps) = self.rco_exp(orelse, True)
                orelse_expr = utils.Begin(
                    [ast.Assign([var], value) for (var, value) in orelse_temps],
                    orelse_expr,
                )

                return ast.IfExp(simplified_cond, body_expr, orelse_expr), cond_temps
            case _:
                raise Exception(f"rco_exp: unexpected value: {e}")

    def rco_stmt(self, s: ast.stmt) -> Sequence[ast.stmt]:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [exp])):
                (atm, temps) = self.rco_exp(exp, True)
                return [ast.Assign([var], value) for (var, value) in temps] + [
                    ast.Expr(ast.Call(ast.Name("print"), [atm]))
                ]
            case ast.Expr(expr):
                (atm, temps) = self.rco_exp(expr, False)
                return [ast.Assign([var], value) for (var, value) in temps] + [
                    ast.Expr(atm)
                ]
            case ast.Assign([ast.Name(var)], exp):
                (atm, temps) = self.rco_exp(exp, True)
                return [ast.Assign([var], value) for (var, value) in temps] + [
                    ast.Assign([ast.Name(var)], atm)
                ]
            case ast.If(test, body, orelse):
                (simplified_cond, cond_temps) = self.rco_exp(test, False)
                body = [s for stmt in body for s in self.rco_stmt(stmt)]
                orelse = [s for stmt in orelse for s in self.rco_stmt(stmt)]
                return [
                    *[ast.Assign([var], value) for (var, value) in cond_temps],
                    ast.If(simplified_cond, body, orelse),
                ]
            case _:
                raise Exception(f"rco_stmt: unexpected stmt: {s}")

    def remove_complex_operands(self, p: ast.Module) -> ast.Module:
        match p:
            case ast.Module(ss):
                sss = [stmt for s in ss for stmt in self.rco_stmt(s)]
                return ast.Module(sss)
        raise Exception("remove_complex_operands not implemented")

    ##### explicate control

    def create_block(
        self, stmts: Sequence[ast.stmt], basic_blocks: dict[str, Sequence[ast.stmt]]
    ) -> Sequence[ast.stmt]:
        match stmts:
            case [utils.Goto(_)]:
                return stmts
            case _:
                label = utils.label_name(utils.generate_name("block"))
                basic_blocks[label] = stmts
                return [utils.Goto(label)]

    def explicate_effect(
        self,
        exp: ast.expr,
        cont: Sequence[ast.stmt],
        basic_blocks: dict[str, Sequence[ast.stmt]],
    ) -> Sequence[ast.stmt]:
        match exp:
            case ast.IfExp(condition, body, orelse):
                body = self.explicate_effect(body, cont, basic_blocks)
                orelse = self.explicate_effect(orelse, cont, basic_blocks)
                return [
                    *self.explicate_pred(condition, body, orelse, basic_blocks),
                    *cont,
                ]
            case ast.Call(name, args):
                return [ast.Expr(ast.Call(name, args)), *cont]
            case utils.Begin(stmts, _):
                return [*stmts, *cont]
            case _:
                return cont

    def explicate_assign(
        self,
        left: ast.Name,
        exp: ast.expr,
        cont: Sequence[ast.stmt],
        basic_blocks: dict[str, Sequence[ast.stmt]],
    ) -> Sequence[ast.stmt]:
        match exp:
            case ast.IfExp(condition, body, orelse):
                body = self.explicate_assign(left, body, cont, basic_blocks)
                orelse = self.explicate_assign(left, orelse, cont, basic_blocks)
                return self.explicate_pred(condition, body, orelse, basic_blocks)
            case utils.Begin(stmts, result):
                return [*stmts, ast.Assign([left], result), *cont]
            case _:
                return [ast.Assign([left], exp), *cont]

    def explicate_pred(
        self,
        cond: ast.expr,
        body: Sequence[ast.stmt],
        orelse: Sequence[ast.stmt],
        basic_blocks: dict[str, Sequence[ast.stmt]],
    ) -> Sequence[ast.stmt]:
        match cond:
            case ast.Compare() | ast.UnaryOp(ast.Not) | ast.Name():
                body = self.create_block(body, basic_blocks)
                orelse = self.create_block(orelse, basic_blocks)
                return self.create_block([ast.If(cond, body, orelse)], basic_blocks)
            case ast.IfExp(condition, cond_body, cond_orelse):
                body = self.create_block(body, basic_blocks)
                orelse = self.create_block(orelse, basic_blocks)

                cond_body = self.explicate_pred(cond_body, body, orelse, basic_blocks)
                cond_orelse = self.explicate_pred(
                    cond_orelse, body, orelse, basic_blocks
                )

                return self.create_block(
                    [ast.If(condition, cond_body, cond_orelse)], basic_blocks
                )
            case ast.Constant(True):
                return body
            case ast.Constant(False):
                return orelse
            case utils.Begin(stmts, result):
                return self.create_block(
                    [
                        *stmts,
                        ast.If(
                            result,
                            self.create_block(body, basic_blocks),
                            self.create_block(orelse, basic_blocks),
                        ),
                    ],
                    basic_blocks,
                )
            case _:
                raise Exception(f"explicate_pred: invalid condition: {cond}")

    def explicate_stmt(
        self,
        statement: ast.stmt,
        cont: Sequence[ast.stmt],
        basic_blocks: dict[str, Sequence[ast.stmt]],
    ) -> Sequence[ast.stmt]:
        match statement:
            case ast.Assign([ast.Name(_) as var], exp):
                return self.explicate_assign(var, exp, cont, basic_blocks)
            case ast.Expr(expr):
                return self.explicate_effect(expr, cont, basic_blocks)
            case ast.If(test, body, orelse):
                compiled_body = self.create_block(
                    [
                        s
                        for stmt in body
                        for s in self.explicate_stmt(stmt, cont, basic_blocks)
                    ],
                    basic_blocks,
                )
                compiled_orelse = self.create_block(
                    [
                        s
                        for stmt in orelse
                        for s in self.explicate_stmt(stmt, cont, basic_blocks)
                    ],
                    basic_blocks,
                )
                return self.explicate_pred(
                    test, compiled_body, compiled_orelse, basic_blocks
                )
            case _:
                raise Exception("explicate_control: invalid statement")

    def explicate_control(self, p: ast.Module):
        match p:
            case ast.Module(body):
                new_body = [ast.Return(ast.Constant(0))]
                basic_blocks: dict[str, Sequence[ast.stmt]] = {}
                for s in reversed(body):
                    new_body = self.explicate_stmt(s, new_body, basic_blocks)
                basic_blocks[utils.label_name("start")] = new_body
                return utils.CProgram(basic_blocks)
