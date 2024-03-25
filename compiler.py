import ast
from typing import MutableMapping, Sequence, Tuple

import utils
import x86_ast

Binding = Tuple[ast.Name, ast.expr]
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

    def select_arg(self, e: ast.expr) -> x86_ast.arg:
        match e:
            case ast.Constant(c):
                return x86_ast.Immediate(c)
            case ast.Name(v):
                return x86_ast.Variable(v)
        raise Exception()

    def select_stmt(self, s: ast.stmt) -> Sequence[x86_ast.instr]:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [atm])):
                arg = self.select_arg(atm)
                return [
                    x86_ast.Instr("movq", [arg, x86_ast.Reg("rdi")]),
                    x86_ast.Callq(utils.label_name("print_int"), 1),
                ]
            case ast.Assign([var], exp):
                arg = self.select_arg(var)
                match exp:
                    case ast.Constant(_) | ast.Name(_):
                        return [x86_ast.Instr("movq", [self.select_arg(exp), arg])]
                    case ast.Call(ast.Name("input_int"), []):
                        return [
                            x86_ast.Callq(utils.label_name("read_int"), 0),
                            x86_ast.Instr("movq", [x86_ast.Reg("rax"), arg]),
                        ]
                    case ast.UnaryOp(ast.USub(), atm):
                        return [
                            x86_ast.Instr("movq", [self.select_arg(atm), arg]),
                            x86_ast.Instr("negq", [arg]),
                        ]
                    case ast.BinOp(atm1, op, atm2):
                        return [
                            x86_ast.Instr("movq", [self.select_arg(atm1), arg]),
                            x86_ast.Instr(
                                "addq" if isinstance(op, ast.Add) else "subq",
                                [self.select_arg(atm2), arg],
                            ),
                        ]
        raise Exception()

    def select_instructions(self, p: ast.Module) -> x86_ast.X86Program:
        match p:
            case ast.Module(stmts):
                instrs = [instr for stmt in stmts for instr in self.select_stmt(stmt)]
                return x86_ast.X86Program(instrs)
        raise Exception()

    ############################################################################
    # Assign Homes
    ############################################################################

    def assign_homes_arg(
        self,
        a: x86_ast.arg,
        home: MutableMapping[x86_ast.Variable, x86_ast.Reg | x86_ast.Deref],
    ) -> x86_ast.arg:
        match a:
            case x86_ast.Variable(_):
                if a not in home:
                    var_count = len(home)
                    home[a] = x86_ast.Deref("rbp", -8 * (var_count + 1))

                return home[a]
            case _:
                return a

    def assign_homes_instr(
        self,
        i: x86_ast.instr,
        home: MutableMapping[x86_ast.Variable, x86_ast.Reg | x86_ast.Deref],
    ) -> x86_ast.instr:
        match i:
            case x86_ast.Instr(op, [arg_0, arg_1]):
                arg_0 = self.assign_homes_arg(arg_0, home)
                arg_1 = self.assign_homes_arg(arg_1, home)
                return x86_ast.Instr(op, [arg_0, arg_1])

            case x86_ast.Instr(op, [arg_0]):
                arg_1 = self.assign_homes_arg(arg_0, home)
                return x86_ast.Instr(op, [arg_0, arg_1])

            case _:
                raise Exception(f"invalid instruction: {i}")

    def assign_homes(self, p: x86_ast.X86Program) -> x86_ast.X86Program:
        homes = {}
        body: list[x86_ast.instr] = []
        for instr in p.body:
            match instr:
                case x86_ast.Instr():
                    body.append(self.assign_homes_instr(instr, homes))
                case _:
                    body.append(instr)  # type: ignore

        frame_size = len(homes) if len(homes) % 2 == 0 else len(homes) + 1
        body = [
            x86_ast.Instr(
                "subq", [x86_ast.Immediate(frame_size * 8), x86_ast.Reg("rsp")]
            ),
            *body,
            x86_ast.Instr(
                "addq", [x86_ast.Immediate(frame_size * 8), x86_ast.Reg("rsp")]
            ),
        ]

        return x86_ast.X86Program(body=body)

    #############################################################################
    ## Patch Instructions
    #############################################################################

    def patch_instr(self, i: x86_ast.instr) -> list[x86_ast.instr]:
        match i:
            case x86_ast.Instr(
                instr,
                [
                    x86_ast.Deref() as arg_0,
                    x86_ast.Deref() as arg_1,
                ],
            ):
                return [
                    x86_ast.Instr("movq", [arg_0, x86_ast.Reg("rax")]),
                    x86_ast.Instr(instr, [x86_ast.Reg("rax"), arg_1]),
                ]
            case _:
                return [i]

    def patch_instructions(self, p: x86_ast.X86Program) -> x86_ast.X86Program:
        body = []
        for instr in p.body:
            match instr:
                case x86_ast.Instr(_, [_, _]):
                    body.extend(self.patch_instr(instr))
                case _:
                    body.append(instr)

        p.body = body
        return p

    #############################################################################
    ## Prelude & Conclusion
    #############################################################################

    def prelude_and_conclusion(self, p: x86_ast.X86Program) -> x86_ast.X86Program:
        body = [
            x86_ast.Instr("pushq", [x86_ast.Reg("rbp")]),
            x86_ast.Instr("movq", [x86_ast.Reg("rsp"), x86_ast.Reg("rbp")]),
            *p.body,
            x86_ast.Instr("popq", [x86_ast.Reg("rbp")]),
            x86_ast.Instr("retq", []),
        ]

        return x86_ast.X86Program(body=body)
