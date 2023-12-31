import ast
from ast import *
from enum import EnumCheck
from utils import *
import utils
from x86_ast import *
import os
from typing import List, Sequence, Tuple, Set, Dict

Binding = Tuple[Name, expr]
Temporaries = Sequence[Binding]


class Compiler:
    ############################################################################
    # Remove Complex Operands
    ############################################################################

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
            case ast.UnaryOp(ast.USub(), exp):
                (atm, temps) = self.rco_exp(exp, True)
                if need_atomic:
                    tmp = utils.generate_name("tmp")
                    return (
                        ast.Name(tmp),
                        [*temps, (ast.Name(tmp), ast.UnaryOp(ast.USub(), atm))],
                    )
                return (ast.UnaryOp(ast.USub(), atm), temps)
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
            case _:
                raise Exception(f"rco_stmt: unexpected stmt: {s}")

    def remove_complex_operands(self, p: ast.Module) -> ast.Module:
        match p:
            case ast.Module(ss):
                sss = [stmt for s in ss for stmt in self.rco_stmt(s)]
                return ast.Module(sss)
        raise Exception("remove_complex_operands not implemented")

    ############################################################################
    # Select Instructions
    ############################################################################

    def select_arg(self, e: expr) -> arg:
        # YOUR CODE HERE
        pass

    def select_stmt(self, s: stmt) -> List[instr]:
        # YOUR CODE HERE
        pass

    def select_instructions(self, p: Module) -> X86Program:
        # YOUR CODE HERE
        pass

    ############################################################################
    # Assign Homes
    ############################################################################

    def assign_homes_arg(self, a: arg, home: Dict[Variable, arg]) -> arg:
        # YOUR CODE HERE
        pass

    def assign_homes_instr(self, i: instr, home: Dict[Variable, arg]) -> instr:
        # YOUR CODE HERE
        pass

    def assign_homes(self, p: X86Program) -> X86Program:
        # YOUR CODE HERE
        pass

    ############################################################################
    # Patch Instructions
    ############################################################################

    def patch_instr(self, i: instr) -> List[instr]:
        # YOUR CODE HERE
        pass

    def patch_instructions(self, p: X86Program) -> X86Program:
        # YOUR CODE HERE
        pass

    ############################################################################
    # Prelude & Conclusion
    ############################################################################

    def prelude_and_conclusion(self, p: X86Program) -> X86Program:
        # YOUR CODE HERE
        pass
