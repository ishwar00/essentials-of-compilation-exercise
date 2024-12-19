import os
import sys

import type_check_Ctup
import type_check_Carray

sys.path.append("../python-student-support-code")
sys.path.append("../python-student-support-code/interp_x86")

# import compiler_Lif as compiler
# import compiler_Lwhile as compiler
# import compiler_Ltup as compiler
import compiler_Larray as compiler

# import interp_Lvar
# import interp_Lif
# import interp_Lwhile
import interp_Ltup
import interp_Larray

# import type_check_Lif
# import type_check_Lwhile
import type_check_Ltup
import type_check_Larray
from utils import run_tests, run_one_test, enable_tracing
from interp_x86.eval_x86 import interp_x86
from interp_Cif import InterpCif
from interp_Ctup import InterpCtup
from interp_Carray import InterpCarray


if len(sys.argv) > 1 and sys.argv[1] == "-v":
    enable_tracing()


class DummyCompile: ...


compiler = compiler.Compiler()
# compiler = DummyCompile()

# typechecker = type_check_Lif.TypeCheckLif().type_check
# typechecker = type_check_Lwhile.TypeCheckLwhile().type_check
# typechecker = type_check_Ltup.TypeCheckLtup().type_check
# typechecker_c = type_check_Ctup.TypeCheckCtup().type_check
typechecker = type_check_Larray.TypeCheckLarray().type_check
typechecker_c = type_check_Carray.TypeCheckCarray().type_check

typecheck_dict = {
    # "source": typechecker,
    "resolve": typechecker,
    "remove_complex_operands": typechecker,
    "explicate_control": typechecker_c
}
# interpreter = interp_Lvar.InterpLvar().interp
# interpreter = interp_Lif.InterpLif().interp
# interpreter = interp_Lwhile.InterpLwhile().interp
# interpreter = interp_Ltup.InterpLtup().interp
interpreter = interp_Larray.InterpLarray().interp
interp_dict = {
    "resolve": interpreter,
    "remove_complex_operands": interpreter,
    "shrink": interpreter,
    "explicate_control": InterpCtup().interp,
    # "expose_allocation": interpreter,
    # "select_instructions": interp_x86,
    # "assign_homes": interp_x86,
    # "patch_instructions": interp_x86,
    "prelude_and_conclusion": interp_x86
}

if True:
    # run_tests('var', compiler, 'var',
    #           typecheck_dict,
    #           interp_dict)
    # run_tests("if", compiler, "if", typecheck_dict, interp_dict)
    # run_tests("while", compiler, "while", typecheck_dict, interp_dict)
    run_tests("array", compiler, "tup", typecheck_dict, interp_dict)
    run_tests("tup", compiler, "tup", typecheck_dict, interp_dict)
    run_tests("var", compiler, "tup", typecheck_dict, interp_dict)
    run_tests("if", compiler, "tup", typecheck_dict, interp_dict)
    run_tests("while", compiler, "tup", typecheck_dict, interp_dict)
else:
    run_one_test(
        os.getcwd() + "/tests/var/zero.py",
        "var",
        compiler,
        "var",
        typecheck_dict,
        interp_dict,
    )
