import ast
from type_check_Lvar import TypeCheckLvar
import utils


class TypeCheckLif(TypeCheckLvar):

    # The following is not needed until Larray, for bounds checking,
    # but I don't want to have to redefine the type checking code
    # for IfExp and If. -eremy
    def combine_types(self, t1, t2):
        match (t1, t2):
            case (utils.Bottom(), _):
                return t2
            case (_, utils.Bottom()):
                return t1
            case _:
                return t1

    def type_check_exp(self, e, env):
        match e:
            case ast.Constant(value) if isinstance(value, bool):
                return utils.BoolType()
            case ast.IfExp(test, body, orelse):
                test_t = self.type_check_exp(test, env)
                self.check_type_equal(utils.BoolType(), test_t, test)
                body_t = self.type_check_exp(body, env)
                orelse_t = self.type_check_exp(orelse, env)
                self.check_type_equal(body_t, orelse_t, e)
                return self.combine_types(body_t, orelse_t)
            case ast.UnaryOp(ast.Not(), v):
                t = self.type_check_exp(v, env)
                self.check_type_equal(t, utils.BoolType(), v)
                return utils.BoolType()
            case ast.BoolOp(_op, values):
                left = values[0]
                right = values[1]
                l = self.type_check_exp(left, env)
                self.check_type_equal(l, utils.BoolType(), left)
                r = self.type_check_exp(right, env)
                self.check_type_equal(r, utils.BoolType(), right)
                return utils.BoolType()
            case ast.Compare(left, [ast.Eq | ast.NotEq], [right]):
                l = self.type_check_exp(left, env)
                r = self.type_check_exp(right, env)
                self.check_type_equal(l, r, e)
                return utils.BoolType()
            case ast.Compare(left, [_], [right]):
                l = self.type_check_exp(left, env)
                self.check_type_equal(l, utils.IntType(), left)
                r = self.type_check_exp(right, env)
                self.check_type_equal(r, utils.IntType(), right)
                return utils.BoolType()
            # case Let(Name(x), rhs, body):
            #   t = self.type_check_exp(rhs, env)
            #   new_env = dict(env); new_env[x] = t
            #   return self.type_check_exp(body, new_env)
            case utils.Begin(ss, e):
                self.type_check_stmts(ss, env)
                return self.type_check_exp(e, env)
            case _:
                return super().type_check_exp(e, env)

    def type_check_stmts(self, ss, env):
        if len(ss) == 0:
            return
        match ss[0]:
            case ast.If(test, body, orelse):
                test_t = self.type_check_exp(test, env)
                self.check_type_equal(utils.BoolType(), test_t, test)
                body_t = self.type_check_stmts(body, env)
                orelse_t = self.type_check_stmts(orelse, env)
                # disabled the following check to experiment with allowing early returns
                # from a functions -Jeremy
                # self.check_type_equal(body_t, orelse_t, ss[0])
                if len(ss) > 1:
                    return self.type_check_stmts(ss[1:], env)
                else:  # this 'if' statement is in tail position
                    return self.combine_types(body_t, orelse_t)
            case _:
                return super().type_check_stmts(ss, env)
