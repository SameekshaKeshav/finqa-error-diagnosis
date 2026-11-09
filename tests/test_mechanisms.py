from finqadiag.dsl import parse_program
from finqadiag.mechanisms import mechanism

G = parse_program("subtract(10, 4), divide(#0, 4)")


def m(p):
    return mechanism(parse_program(p), G)


def test_categories():
    assert mechanism(None, G) == "unparsable"
    assert m("subtract(10, 4)") == "too_few_steps"
    assert m("subtract(10, 4), divide(#0, 4), add(#1, 1)") == "too_many_steps"
    assert m("add(10, 4), divide(#0, 4)") == "wrong_operator"
    assert m("subtract(4, 10), divide(#0, 4)") == "args_swapped"
    assert m("subtract(10, 4), divide(#0, const_100)") == "wrong_constant"
    assert m("subtract(10, 4), divide(4, #0)") == "args_swapped"
    assert m("subtract(10, 5), divide(#0, 4)") == "wrong_operand"
    assert m("subtract(10, 4), divide(#0, #0)") == "wrong_reference"
