import os
import sys

sys.path.append("../python-student-support-code")
sys.path.append("../python-student-support-code/interp_x86")

import compiler_Lif as compiler

# import interp_Lvar
import interp_Lif
import type_check_Lif
from utils import run_tests, run_one_test, enable_tracing
from interp_x86.eval_x86 import interp_x86
from interp_Cif import InterpCif


enable_tracing()


class DummyCompile: ...


compiler = compiler.Compiler()
# compiler = DummyCompile()

typechecker = type_check_Lif.TypeCheckLif().type_check

typecheck_dict = {
    "source": typechecker,
    "remove_complex_operands": typechecker,
}
# interpreter = interp_Lvar.InterpLvar().interp
interpreter = interp_Lif.InterpLif().interp
interp_dict = {
    "remove_complex_operands": interpreter,
    "shrink": interpreter,
    "select_instructions": interp_x86,
    "assign_homes": interp_x86,
    "patch_instructions": interp_x86,
    "explicate_control": InterpCif().interp,
}

if True:
    # run_tests('var', compiler, 'var',
    #           typecheck_dict,
    #           interp_dict)
    run_tests("if", compiler, "if", typecheck_dict, interp_dict)
else:
    run_one_test(
        os.getcwd() + "/tests/var/zero.py",
        "var",
        compiler,
        "var",
        typecheck_dict,
        interp_dict,
    )
