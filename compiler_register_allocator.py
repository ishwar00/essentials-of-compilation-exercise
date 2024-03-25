import itertools

import compiler
import x86_ast
from graph import UndirectedAdjList

# Skeleton code for the chapter on Register Allocation

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


class Compiler(compiler.Compiler):
    ###########################################################################
    # Uncover Live
    ###########################################################################

    @staticmethod
    def _location_set(arg: x86_ast.arg) -> set[x86_ast.location]:
        if isinstance(arg, (x86_ast.Reg, x86_ast.Variable)):
            return {arg}

        return set()

    def read_vars(self, i: x86_ast.instr) -> set[x86_ast.location]:
        match i:
            case x86_ast.Instr("addq" | "subq", [arg_0, arg_1]):
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

    def uncover_live(
        self, p: x86_ast.X86Program
    ) -> dict[x86_ast.instr, set[x86_ast.location]]:
        assert isinstance(p.body, list)

        live_after_set: dict[x86_ast.instr, set[x86_ast.location]] = {p.body[-1]: set()}
        next_after_set = set()
        for instr, next_instr in zip(reversed(p.body[:-1]), reversed(p.body)):
            next_bofore_set = (
                next_after_set - self.write_vars(next_instr)
            ) | self.read_vars(next_instr)
            live_after_set[instr] = next_bofore_set
            next_after_set = next_bofore_set

        return live_after_set

    ############################################################################
    # Build Interference
    ############################################################################

    def build_interference(
        self,
        p: x86_ast.X86Program,
        live_after: dict[x86_ast.instr, set[x86_ast.location]],
    ) -> UndirectedAdjList:
        graph = UndirectedAdjList()

        assert isinstance(p.body, list)
        for instr in p.body:
            write_locations = self.write_vars(instr)
            live_after_set = live_after[instr]

            match instr:
                case x86_ast.Instr("movq", [s, d]):
                    for live_location in live_after_set:
                        if d != live_location and s != live_location:
                            graph.add_edge(d, live_location)
                case _:
                    for write_location in write_locations:
                        for live_location in live_after_set:
                            if write_location != live_location:
                                graph.add_edge(write_location, live_location)

        return graph

    ############################################################################
    # Allocate Registers
    ############################################################################

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
        print("\n", dot_graph)
        dot_graph.render(outfile="graph.pdf")
        for vertex in interference_graph.vertices():
            if isinstance(vertex, x86_ast.Reg):
                if vertex in _register_to_color:
                    color = _register_to_color[vertex]

                    for edge in interference_graph.out_edges(vertex):
                        neighbour = edge.target
                        saturation_set.get(neighbour, set()).add(color)

        print(f"{saturation_set=}")
        print(f"{variables=}")

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

            print(f"{chosen_sat_var=} {reg_allocation[chosen_sat_var]=}")

            for edge in interference_graph.out_edges(chosen_sat_var):
                if edge.target in saturation_set:
                    saturation_set[edge.target].add(reg_allocation[chosen_sat_var])
                    print(f"{saturation_set=}")

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

        assert isinstance(p.body, list)
        for instr in p.body:
            match instr:
                case x86_ast.Instr(
                    "movq", [x86_ast.Variable(_) as s, x86_ast.Variable(_) as d]
                ):
                    graph.add_edge(s, d)

        return graph

    ############################################################################
    # Assign Homes
    ############################################################################

    def assign_homes(self, p: x86_ast.X86Program) -> x86_ast.X86Program:
        live_after_set = self.uncover_live(p)
        graph = self.build_interference(p, live_after_set)
        move_graph = self.build_move_graph(p)
        reg_allocation, spilled_count = self.allocate_registers(graph, move_graph)

        body: list[x86_ast.instr] = []
        for instr in p.body:
            match instr:
                case x86_ast.Instr():
                    body.append(self.assign_homes_instr(instr, reg_allocation))
                case _:
                    body.append(instr)  # type: ignore

        used_locations = set(reg_allocation.values())
        used_callee = _callee_saved_registers & used_locations

        return x86_ast.X86Program(
            body=body, spilled_count=spilled_count, used_callee=used_callee
        )

    ###########################################################################
    # Patch Instructions
    ###########################################################################

    def patch_instr(self, i: x86_ast.instr) -> list[x86_ast.instr]:
        match i:
            case x86_ast.Instr(
                instr,
                [arg_0, arg_1],
            ) if arg_0 == arg_1:
                return []

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

    ###########################################################################
    # Prelude & Conclusion
    ###########################################################################

    def prelude_and_conclusion(self, p: x86_ast.X86Program) -> x86_ast.X86Program:
        assert p.spilled_count is not None
        assert p.used_callee is not None

        total_used = p.spilled_count + len(p.used_callee)

        frame_size = (total_used if total_used % 2 == 0 else total_used + 1) - len(
            p.used_callee
        )
        body = [
            x86_ast.Instr("pushq", [x86_ast.Reg("rbp")]),
            x86_ast.Instr("movq", [x86_ast.Reg("rsp"), x86_ast.Reg("rbp")]),
            x86_ast.Instr(
                "subq", [x86_ast.Immediate(frame_size * 8), x86_ast.Reg("rsp")]
            ),
            *(x86_ast.Instr("pushq", [r]) for r in p.used_callee),
            *p.body,
            *(x86_ast.Instr("popq", [r]) for r in p.used_callee),
            x86_ast.Instr(
                "addq", [x86_ast.Immediate(frame_size * 8), x86_ast.Reg("rsp")]
            ),
            x86_ast.Instr("popq", [x86_ast.Reg("rbp")]),
            x86_ast.Instr("retq", []),
        ]

        return x86_ast.X86Program(body=body)
