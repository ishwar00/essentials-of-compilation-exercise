import os
import sys

sys.path.append("../python-student-support-code")
sys.path.append("../python-student-support-code/interp_x86")

# import compiler_Lif as compiler
# import compiler_Lwhile as compiler
import compiler_Ltup as compiler

# import interp_Lvar
# import interp_Lif
# import interp_Lwhile
import interp_Ltup

# import type_check_Lif
# import type_check_Lwhile
import type_check_Ltup
from utils import run_tests, run_one_test, enable_tracing
from interp_x86.eval_x86 import interp_x86
from interp_Cif import InterpCif


if len(sys.argv) > 1 and sys.argv[1] == "-v":
    enable_tracing()


class DummyCompile: ...


compiler = compiler.Compiler()
# compiler = DummyCompile()

# typechecker = type_check_Lif.TypeCheckLif().type_check
# typechecker = type_check_Lwhile.TypeCheckLwhile().type_check
typechecker = type_check_Ltup.TypeCheckLtup().type_check

typecheck_dict = {
    "source": typechecker,
    "remove_complex_operands": typechecker,
}
# interpreter = interp_Lvar.InterpLvar().interp
# interpreter = interp_Lif.InterpLif().interp
# interpreter = interp_Lwhile.InterpLwhile().interp
interpreter = interp_Ltup.InterpLtup().interp
interp_dict = {
    "remove_complex_operands": interpreter,
    "shrink": interpreter,
    "select_instructions": interp_x86,
    "assign_homes": interp_x86,
    "patch_instructions": interp_x86,
    "explicate_control": InterpCif().interp,
    "expose_allocation": interpreter,
}

if True:
    # run_tests('var', compiler, 'var',
    #           typecheck_dict,
    #           interp_dict)
    # run_tests("if", compiler, "if", typecheck_dict, interp_dict)
    # run_tests("while", compiler, "while", typecheck_dict, interp_dict)
    run_tests("tup", compiler, "tup", typecheck_dict, interp_dict)
else:
    run_one_test(
        os.getcwd() + "/tests/var/zero.py",
        "var",
        compiler,
        "var",
        typecheck_dict,
        interp_dict,
    )
