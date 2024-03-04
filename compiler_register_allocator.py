import compiler
import x86_ast
from graph import UndirectedAdjList

# Skeleton code for the chapter on Register Allocation


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
        self, graph: UndirectedAdjList, variables: set[x86_ast.Variable]
    ) -> dict[x86_ast.Variable, int]:
        reg_allocation: dict[x86_ast.Variable, int] = {}
        # TODO: use priority queue
        saturation_set: dict[x86_ast.Variable, set[int]] = {
            variable: set() for variable in variables
        }
        while len(variables) > 0:
            max_saturation = max(len(value) for value in saturation_set.values())
            most_sat_var = [
                key
                for key, value in saturation_set.items()
                if len(value) == max_saturation
            ][0]
            reg_allocation[most_sat_var] = len(saturation_set[most_sat_var])

            for edge in graph.out_edges(most_sat_var):
                if edge.target in saturation_set:
                    saturation_set[edge.target].add(reg_allocation[most_sat_var])

            saturation_set.pop(most_sat_var)
            variables.remove(most_sat_var)

        return reg_allocation

    def allocate_registers(
        self, graph: UndirectedAdjList
    ) -> tuple[dict[x86_ast.Variable, x86_ast.Deref | x86_ast.Reg], int]:
        variables: set[x86_ast.Variable] = {
            vertex
            for vertex in graph.vertices()
            if isinstance(vertex, x86_ast.Variable)
        }
        color_allocation = self.color_graph(graph, variables)
        registers = {
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

        reg_allocation: dict[x86_ast.Variable, x86_ast.Deref | x86_ast.Reg] = {}
        spilled_count: int = 0
        for loc, color in color_allocation.items():
            if color in registers:
                reg_allocation[loc] = registers[color]
            else:
                offset = color - len(registers)
                reg_allocation[loc] = x86_ast.Deref("rbp", -8 * offset)
                spilled_count = max(offset, spilled_count)

        return reg_allocation, spilled_count

    ############################################################################
    # Assign Homes
    ############################################################################

    def assign_homes(self, p: x86_ast.X86Program) -> x86_ast.X86Program:
        live_after_set = self.uncover_live(p)
        graph = self.build_interference(p, live_after_set)
        reg_allocation, spilled_count = self.allocate_registers(graph)

        body: list[x86_ast.instr] = []
        for instr in p.body:
            match instr:
                case x86_ast.Instr():
                    body.append(self.assign_homes_instr(instr, reg_allocation))
                case _:
                    body.append(instr)  # type: ignore

        frame_size = spilled_count if spilled_count % 2 == 0 else spilled_count + 1
        body = [
            x86_ast.Instr('subq', [x86_ast.Immediate(frame_size * 8), x86_ast.Reg('rsp')]),
            *body,
            x86_ast.Instr('addq', [x86_ast.Immediate(frame_size * 8), x86_ast.Reg('rsp')]),
        ]

        return x86_ast.X86Program(body=body)

    ###########################################################################
    # Patch Instructions
    ###########################################################################

    # def patch_instructions(self, p: X86Program) -> X86Program:
    #     # YOUR CODE HERE
    #     pass

    ###########################################################################
    # Prelude & Conclusion
    ###########################################################################

    # def prelude_and_conclusion(self, p: X86Program) -> X86Program:
    #     # YOUR CODE HERE
    #     pass
