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
    # def color_graph(self, graph: UndirectedAdjList,
    #                 variables: Set[location]) -> Tuple[Dict[location, int], Set[location]]:
    #     # YOUR CODE HERE
    #     pass

    # def allocate_registers(self, p: X86Program,
    #                        graph: UndirectedAdjList) -> X86Program:
    #     # YOUR CODE HERE
    #     pass

    ############################################################################
    # Assign Homes
    ############################################################################

    # def assign_homes(self, pseudo_x86: X86Program) -> X86Program:
    #     # YOUR CODE HERE
    #     pass

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
