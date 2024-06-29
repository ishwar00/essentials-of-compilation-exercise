import ast
import itertools
from typing import Sequence, Tuple

import utils
import x86_ast
from graph import (DirectedAdjList, UndirectedAdjList, topological_sort,
                   transpose)

Binding = Tuple[ast.Name, ast.expr]
Temporaries = Sequence[Binding]

_color_to_register = {
    0: x86_ast.Reg("rcx"),
    1: x86_ast.Reg("rdx"),
    2: x86_ast.Reg("rsi"),
    3: x86_ast.Reg("rdi"),
    4: x86_ast.Reg("r8"),
    5: x86_ast.Reg("r9"),
    6: x86_ast.Reg("r10"),
    7: x86_ast.Reg("rbx"),
    8: x86_ast.Reg("r12"),
    9: x86_ast.Reg("r13"),
    10: x86_ast.Reg("r14"),
}
_callee_saved_registers = {
    x86_ast.Reg("rbx"),
    x86_ast.Reg("r12"),
    x86_ast.Reg("r13"),
    x86_ast.Reg("r14"),
}
_register_to_color = {reg: color for color, reg in _color_to_register.items()}


def _condition_code(op: ast.cmpop) -> str:
    match op:
        case ast.Lt():
            return "l"
        case ast.LtE():
            return "le"
        case ast.Lt():
            return "l"
        case ast.Gt():
            return "g"
        case ast.GtE():
            return "ge"
        case ast.NotEq():
            return "ne"
        case ast.Eq():
            return "e"
        case _:
            raise Exception(f"_condition_code: unexpected op: {op}. type: {type(op)}")


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

                (body_expr, body_temps) = self.rco_exp(body, need_atomic)
                if body_temps:
                    body_expr = utils.Begin(
                        [ast.Assign([var], value) for (var, value) in body_temps],
                        body_expr,
                    )

                (orelse_expr, orelse_temps) = self.rco_exp(orelse, need_atomic)
                if orelse_temps:
                    orelse_expr = utils.Begin(
                        [ast.Assign([var], value) for (var, value) in orelse_temps],
                        orelse_expr,
                    )

                if need_atomic:
                    tmp = utils.generate_name("tmp")
                    return (
                        ast.Name(tmp),
                        [
                            *cond_temps,
                            (
                                ast.Name(tmp),
                                ast.IfExp(simplified_cond, body_expr, orelse_expr),
                            ),
                        ],
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
            case ast.Compare() | ast.UnaryOp(ast.Not()) | ast.Name():
                return self.create_block(
                    [
                        ast.If(
                            cond,
                            self.create_block(body, basic_blocks),
                            self.create_block(orelse, basic_blocks),
                        )
                    ],
                    basic_blocks,
                )
            case utils.Begin(stmts, result):
                return self.create_block(
                    [
                        *stmts,
                        *self.explicate_pred(result, body, orelse, basic_blocks),
                    ],
                    basic_blocks,
                )
            case ast.IfExp(condition, cond_body, cond_orelse):
                body = self.create_block(body, basic_blocks)
                orelse = self.create_block(orelse, basic_blocks)

                cond_body = self.explicate_pred(cond_body, body, orelse, basic_blocks)
                cond_orelse = self.explicate_pred(
                    cond_orelse, body, orelse, basic_blocks
                )

                return self.explicate_pred(
                    condition, cond_body, cond_orelse, basic_blocks
                )
            case ast.Constant(True):
                return body
            case ast.Constant(False):
                return orelse
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
                        for stmt in body[:-1]
                        for s in self.explicate_stmt(stmt, [], basic_blocks)
                    ]
                    # the continuation needs to be passed only to the last statement
                    + list(self.explicate_stmt(body[-1], cont, basic_blocks)),
                    basic_blocks,
                )
                compiled_orelse = self.create_block(
                    [
                        s
                        for stmt in orelse[:-1]
                        for s in self.explicate_stmt(stmt, [], basic_blocks)
                    ]
                    # the continuation needs to be passed only to the last statement
                    + list(self.explicate_stmt(orelse[-1], cont, basic_blocks)),
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

    ### select instructions

    def select_arg(self, e: ast.expr) -> x86_ast.arg:
        match e:
            case ast.Constant(True):
                return x86_ast.Immediate(1)
            case ast.Constant(False):
                return x86_ast.Immediate(0)
            case ast.Constant(c):
                return x86_ast.Immediate(c)
            case ast.Name(v):
                return x86_ast.Variable(v)
        raise Exception()

    def select_stmt(self, s: ast.stmt) -> Sequence[x86_ast.instr]:
        match s:
            case ast.Expr(ast.Call(ast.Name("print"), [atm])):
                lhs = self.select_arg(atm)
                return [
                    x86_ast.Instr("movq", [lhs, x86_ast.Reg("rdi")]),
                    x86_ast.Callq(utils.label_name("print_int"), 1),
                ]
            case ast.Assign([var], exp):
                lhs = self.select_arg(var)
                match exp:
                    case ast.Constant(_) | ast.Name(_):
                        return [x86_ast.Instr("movq", [self.select_arg(exp), lhs])]
                    case ast.Call(ast.Name("input_int"), []):
                        return [
                            x86_ast.Callq(utils.label_name("read_int"), 0),
                            x86_ast.Instr("movq", [x86_ast.Reg("rax"), lhs]),
                        ]
                    case ast.UnaryOp(ast.USub(), atm):
                        return [
                            x86_ast.Instr("movq", [self.select_arg(atm), lhs]),
                            x86_ast.Instr("negq", [lhs]),
                        ]
                    case ast.BinOp(atm1, op, atm2):
                        return [
                            x86_ast.Instr("movq", [self.select_arg(atm1), lhs]),
                            x86_ast.Instr(
                                "addq" if isinstance(op, ast.Add) else "subq",
                                [self.select_arg(atm2), lhs],
                            ),
                        ]
                    case ast.UnaryOp(ast.Not(), atm):
                        match atm:
                            case ast.Name(left_var) if isinstance(
                                lhs, x86_ast.Variable
                            ) and left_var == lhs.id:
                                return [
                                    x86_ast.Instr("xorq", [x86_ast.Immediate(1), lhs])
                                ]
                            case _:
                                atm = self.select_arg(atm)
                                return [
                                    x86_ast.Instr("movq", [atm, lhs]),
                                    x86_ast.Instr("xorq", [x86_ast.Immediate(1), lhs]),
                                ]
                    case ast.Compare(atm1, [op], [atm2]):
                        atm1 = self.select_arg(atm1)
                        atm2 = self.select_arg(atm2)
                        return [
                            x86_ast.Instr("cmpq", [atm2, atm1]),
                            x86_ast.Instr(
                                f"set{_condition_code(op)}", [x86_ast.Reg("al")]
                            ),
                            x86_ast.Instr("movzbq", [x86_ast.Reg("al"), lhs]),
                        ]
                    case _:
                        raise Exception(f"Unsupported expression in assignment: {exp}")
            case ast.If(cond, body, orelse):
                assert isinstance(body[0], utils.Goto)
                assert isinstance(orelse[0], utils.Goto)
                body_label = body[0].label
                orelse_label = orelse[0].label
                match cond:
                    case ast.Compare(left, [op], [right]):
                        left = self.select_arg(left)
                        right = self.select_arg(right)
                        return [
                            x86_ast.Instr("cmpq", [right, left]),
                            x86_ast.JumpIf(_condition_code(op), body_label),
                            x86_ast.Jump(orelse_label),
                        ]
                    case ast.UnaryOp(ast.Not(), exp):
                        exp = self.select_arg(exp)
                        return [
                            x86_ast.Instr("cmpq", [exp, x86_ast.Immediate(0)]),
                            x86_ast.JumpIf("e", body_label),
                            x86_ast.Jump(orelse_label),
                        ]
                    case ast.Name(var):
                        exp = self.select_arg(cond)
                        return [
                            x86_ast.Instr("cmpq", [exp, x86_ast.Immediate(1)]),
                            x86_ast.JumpIf("e", body_label),
                            x86_ast.Jump(orelse_label),
                        ]
                    case _:
                        raise Exception(
                            f"Unsupported if condition: {cond}. type: {type(cond)}"
                        )
            case ast.Return(value):
                value = (
                    self.select_arg(value)
                    if value is not None
                    else x86_ast.Immediate(0)
                )
                return [
                    x86_ast.Instr("movq", [value, x86_ast.Reg("rax")]),
                    x86_ast.Jump(x86_ast.label_name("conclusion")),
                ]
            case utils.Goto(label):
                return [x86_ast.Jump(label)]
            case _:
                raise Exception(f"Unsupported statement type: {s}")

    def select_instructions(self, p: utils.CProgram) -> x86_ast.X86Program:
        x86_blocks: dict[str, list[x86_ast.instr]] = {}

        for label, stmts in p.body.items():
            x86_blocks[label] = [
                instr for stmt in stmts for instr in self.select_stmt(stmt)
            ]

        return x86_ast.X86Program(x86_blocks)

    @staticmethod
    def _location_set(arg: x86_ast.arg) -> set[x86_ast.location]:
        if isinstance(arg, (x86_ast.Reg, x86_ast.Variable)):
            return {arg}

        return set()

    def read_vars(self, i: x86_ast.instr) -> set[x86_ast.location]:
        match i:
            case x86_ast.Instr("addq" | "subq" | "cmpq", [arg_0, arg_1]):
                return self._location_set(arg_0) | self._location_set(arg_1)
            case x86_ast.Instr("movq", [arg_0, _]):
                return self._location_set(arg_0)
            case x86_ast.Instr("negq" | "pushq", [arg_0]):
                return self._location_set(arg_0)
            case x86_ast.Callq(_, arg_num):
                arg_regs = [
                    x86_ast.Reg("rdi"),
                    x86_ast.Reg("rsi"),
                    x86_ast.Reg("rdx"),
                    x86_ast.Reg("rcx"),
                    x86_ast.Reg("r8"),
                    x86_ast.Reg("r9"),
                ]
                return set(arg_regs[:arg_num])
        return set()

    def write_vars(self, i: x86_ast.instr) -> set[x86_ast.location]:
        match i:
            case x86_ast.Instr("addq" | "subq" | "movq", [_, arg_1]):
                return self._location_set(arg_1)
            case x86_ast.Instr("negq" | "popq", [arg_0]):
                return self._location_set(arg_0)
            case x86_ast.Callq(_, _):
                return {
                    x86_ast.Reg("rax"),
                    x86_ast.Reg("rcx"),
                    x86_ast.Reg("rdx"),
                    x86_ast.Reg("rsi"),
                    x86_ast.Reg("rdi"),
                    x86_ast.Reg("r8"),
                    x86_ast.Reg("r9"),
                    x86_ast.Reg("r10"),
                    x86_ast.Reg("r11"),
                }

        return set()

    def _build_control_flow_graph(self, program: x86_ast.X86Program) -> DirectedAdjList:
        assert isinstance(program.body, dict)
        graph = DirectedAdjList()
        for label, body in program.body.items():
            for instr in body:
                match instr:
                    case x86_ast.Jump(target_label) | x86_ast.JumpIf(_, target_label):
                        graph.add_edge(label, target_label)
                    case _:
                        pass
        return graph

    def uncover_live_in_block(
        self,
        label: str,
        block: list[x86_ast.instr],
        live_before_block: dict[str, set[x86_ast.location]],
    ) -> dict[x86_ast.instr, set[x86_ast.location]]:
        live_after_set: dict[x86_ast.instr, set[x86_ast.location]] = {}

        match block[-1]:
            case x86_ast.Jump(target_label):
                live_after_set[block[-1]] = live_before_block[target_label]
                next_after_set = live_before_block[target_label]
            case _:
                raise Exception(f"invalid last instruction: {block[-1]}")

        for instr, next_instr in zip(reversed(block[:-1]), reversed(block)):
            match instr:
                case x86_ast.Jump(target_label):
                    live_after_set[instr] = live_before_block[target_label]
                case x86_ast.JumpIf(_, target_label):
                    live_before_jump_target = live_before_block[target_label]
                    live_before_next = (
                        next_after_set - self.write_vars(next_instr)
                    ) | self.read_vars(next_instr)
                    live_after_set[instr] = live_before_jump_target | live_before_next
                case _:
                    next_before_set = (
                        next_after_set - self.write_vars(next_instr)
                    ) | self.read_vars(next_instr)
                    live_after_set[instr] = next_before_set
            next_after_set = live_after_set[instr]

        live_before_block[label] = (
            next_after_set - self.write_vars(block[0])
        ) | self.read_vars(block[0])

        return live_after_set

    def uncover_live(
        self, program: x86_ast.X86Program
    ) -> dict[str, dict[x86_ast.instr, set[x86_ast.location]]]:
        assert isinstance(program.body, dict)

        cfg = self._build_control_flow_graph(program)
        cfg = transpose(cfg)
        blocks = topological_sort(cfg)

        live_after_sets: dict[str, dict[x86_ast.instr, set[x86_ast.location]]] = {}
        live_before_block: dict[str, set[x86_ast.location]] = {}

        for label in blocks:
            if label == utils.label_name("conclusion"):
                live_after_sets[label] = {}
                live_before_block[label] = set()
            else:
                block = program.body[label]
                live_after_set = self.uncover_live_in_block(
                    label, block, live_before_block
                )
                live_after_sets[label] = live_after_set
        # breakpoint()

        return live_after_sets

    @staticmethod
    def _handle_byte_reg(loc: x86_ast.arg) -> x86_ast.arg:
        match loc:
            case x86_ast.ByteReg("ah" | "al"):
                return x86_ast.Reg("rax")
            case x86_ast.ByteReg("bh" | "bl"):
                return x86_ast.Reg("rbx")
            case x86_ast.ByteReg("ch" | "cl"):
                return x86_ast.Reg("rcx")
            case x86_ast.ByteReg("dh" | "dl"):
                return x86_ast.Reg("rdx")
        return loc

    def build_interference(
        self,
        p: x86_ast.X86Program,
        live_after: dict[str, dict[x86_ast.instr, set[x86_ast.location]]],
    ) -> UndirectedAdjList:
        graph = UndirectedAdjList()

        assert isinstance(p.body, dict)
        for label, block in p.body.items():
            for instr in block:
                write_locations = self.write_vars(instr)
                live_after_set = live_after[label][instr]

                match instr:
                    case x86_ast.Instr("movq" | "movzbq", [s, d]):
                        assert isinstance(
                            d, x86_ast.location
                        ), f"{d} needs to be a location"
                        s = self._handle_byte_reg(s)
                        d = self._handle_byte_reg(d)
                        for live_location in live_after_set:
                            live_location = self._handle_byte_reg(live_location)
                            if d != live_location and s != live_location:
                                graph.add_edge(d, live_location)
                    case _:
                        for write_location in write_locations:
                            write_location = self._handle_byte_reg(write_location)
                            for live_location in live_after_set:
                                live_location = self._handle_byte_reg(live_location)
                                if write_location != live_location:
                                    graph.add_edge(write_location, live_location)

        return graph

    # # Returns the coloring and the set of spilled variables.
    def color_graph(
        self,
        interference_graph: UndirectedAdjList,
        variables: set[x86_ast.Variable],
        move_graph: UndirectedAdjList,
    ) -> dict[x86_ast.Variable, int]:
        reg_allocation: dict[x86_ast.Variable, int] = {}
        # TODO: use priority queue
        saturation_set: dict[x86_ast.Variable, set[int]] = {
            variable: set() for variable in variables
        }

        dot_graph = interference_graph.show()
        utils.trace(f"\n{dot_graph}")
        dot_graph.render(outfile="graph.pdf")

        for vertex in interference_graph.vertices():
            if isinstance(vertex, x86_ast.Reg):
                if vertex in _register_to_color:
                    color = _register_to_color[vertex]

                    for edge in interference_graph.out_edges(vertex):
                        neighbour = edge.target
                        saturation_set.get(neighbour, set()).add(color)

        utils.trace(f"{saturation_set=}")
        utils.trace(f"{variables=}")

        while len(variables) > 0:
            max_saturation = max(len(value) for value in saturation_set.values())
            most_sat_vars = [
                key
                for key, value in saturation_set.items()
                if len(value) == max_saturation
            ]

            chosen_sat_var: None | x86_ast.Variable = None
            move_related_color: int | None = None
            for var in most_sat_vars:
                if var not in move_graph.out:
                    continue

                for edge in move_graph.out_edges(var):
                    color = reg_allocation.get(edge.target)
                    if color is not None and color not in saturation_set[var]:
                        move_related_color = color
                        chosen_sat_var = var
                        break

                if chosen_sat_var is not None:
                    break

            if chosen_sat_var is None:
                chosen_sat_var = most_sat_vars[0]

            lowest_avail_color = 0
            for color in itertools.count():
                if color not in saturation_set[chosen_sat_var]:
                    lowest_avail_color = color
                    break

            # both stack
            # both are registers
            # lowest_avail_color is register and move_related_color is stack
            if move_related_color in _color_to_register:
                reg_allocation[chosen_sat_var] = move_related_color
            elif (
                move_related_color is not None
                and move_related_color not in _color_to_register
                and lowest_avail_color not in _color_to_register
            ):
                reg_allocation[chosen_sat_var] = move_related_color
            else:
                reg_allocation[chosen_sat_var] = lowest_avail_color

            utils.trace(f"{chosen_sat_var=} {reg_allocation[chosen_sat_var]=}")

            for edge in interference_graph.out_edges(chosen_sat_var):
                if edge.target in saturation_set:
                    saturation_set[edge.target].add(reg_allocation[chosen_sat_var])
                    utils.trace(f"{saturation_set=}")

            saturation_set.pop(chosen_sat_var)
            variables.remove(chosen_sat_var)

        return reg_allocation

    def allocate_registers(
        self, interference_graph: UndirectedAdjList, move_graph: UndirectedAdjList
    ) -> tuple[dict[x86_ast.Variable, x86_ast.Deref | x86_ast.Reg], int]:
        variables: set[x86_ast.Variable] = {
            vertex
            for vertex in interference_graph.vertices()
            if isinstance(vertex, x86_ast.Variable)
        }
        color_allocation = self.color_graph(interference_graph, variables, move_graph)

        reg_allocation: dict[x86_ast.Variable, x86_ast.Deref | x86_ast.Reg] = {}
        spilled_count: int = 0
        for loc, color in color_allocation.items():
            if color in _color_to_register:
                reg_allocation[loc] = _color_to_register[color]
            else:
                offset = color - len(_color_to_register)
                reg_allocation[loc] = x86_ast.Deref("rbp", -8 * offset)
                spilled_count = max(offset, spilled_count)

        return reg_allocation, spilled_count

    def build_move_graph(self, p: x86_ast.X86Program) -> UndirectedAdjList:
        graph = UndirectedAdjList()

        assert isinstance(p.body, dict)
        for block in p.body.values():
            for instr in block:
                match instr:
                    case x86_ast.Instr(
                        "movq", [x86_ast.Variable(_) as s, x86_ast.Variable(_) as d]
                    ):
                        graph.add_edge(s, d)

        return graph

    def assign_homes_arg(
        self,
        a: x86_ast.arg,
        home: dict[x86_ast.Variable, x86_ast.Reg | x86_ast.Deref],
    ) -> x86_ast.arg:
        match a:
            case x86_ast.Variable(_):
                if a not in home:
                    return _color_to_register[5]
                    # raise Exception(f"{a} not in home: {home}")

                return home[a]
            case _:
                return a

    def assign_homes_instr(
        self,
        i: x86_ast.instr,
        home: dict[x86_ast.Variable, x86_ast.Reg | x86_ast.Deref],
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
        live_after_set = self.uncover_live(p)
        graph = self.build_interference(p, live_after_set)
        move_graph = self.build_move_graph(p)
        reg_allocation, spilled_count = self.allocate_registers(graph, move_graph)

        assert isinstance(p.body, dict)

        def _transform_instr(instr: x86_ast.instr) -> x86_ast.instr:
            match instr:
                case x86_ast.Instr():
                    return self.assign_homes_instr(instr, reg_allocation)
                case _:
                    return instr

        new_body = {
            label: [_transform_instr(instr) for instr in block]
            for label, block in p.body.items()
        }

        used_locations = set(reg_allocation.values())
        used_callee = _callee_saved_registers & used_locations

        return x86_ast.X86Program(
            body=new_body, spilled_count=spilled_count, used_callee=used_callee
        )
