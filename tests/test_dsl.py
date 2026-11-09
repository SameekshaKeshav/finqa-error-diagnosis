import pytest

from finqadiag import dsl
from finqadiag.dsl import ExecError, ProgramError, Step

TABLE = [
    ["", "2018", "2017", "2016"],
    ["revenue ( millions )", "$ 100", "$ 80", "$ 60"],
    ["margin", "10% ( 10 % )", "12% ( 12 % )", "14% ( 14 % )"],
    ["margin", "20% ( 20 % )", "22% ( 22 % )", "24% ( 24 % )"],   # duplicate label
]


def test_parse_basic_and_nested_refs():
    steps = dsl.parse_program("subtract(920, 95), subtract(469, 77), subtract(#1, #0), divide(#2, #0)")
    assert [s.op for s in steps] == ["subtract", "subtract", "subtract", "divide"]
    assert steps[2].args == ("#1", "#0")


def test_parse_label_with_parentheses():
    steps = dsl.parse_program("table_sum(revenue ( millions ), none)")
    assert steps == [Step("table_sum", ("revenue ( millions )", "none"))]


@pytest.mark.parametrize("bad", ["", "foo(1, 2)", "add(1)", "add(1, 2", "add(1, 2) divide(3, 4)"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ProgramError):
        dsl.parse_program(bad)


def test_token_roundtrip():
    steps = dsl.parse_program("divide(637, const_5), add(#0, 3)")
    toks = dsl.program_to_tokens(steps)
    assert toks[-1] == dsl.EOF and toks[:6] == ["divide", "(", "637", ",", "const_5", ")"]
    assert dsl.parse_tokens(toks) == steps


@pytest.mark.parametrize("toks", [["divide", "(", "1", ",", "2"], ["divide", "1", "2"], [dsl.EOF], ["bogus", "(", "1", ",", "2", ")"]])
def test_parse_tokens_rejects_malformed(toks):
    with pytest.raises(ProgramError):
        dsl.parse_tokens(toks)


@pytest.mark.parametrize("s,v", [("5", 5.0), ("1,234.5", 1234.5), ("$ 12", 12.0), ("12%", 0.12),
                                 ("( 23 )", -23.0), ("const_100", 100.0), ("const_m1", -1.0), ("-5", -5.0)])
def test_str_to_num(s, v):
    assert dsl.str_to_num(s) == pytest.approx(v)


@pytest.mark.parametrize("s", ["abc", "", "const_", "12abc", "nan", "inf"])
def test_str_to_num_none(s):
    assert dsl.str_to_num(s) is None


def test_execute_arithmetic_and_refs():
    steps = dsl.parse_program("subtract(920, 95), divide(#0, 95)")
    assert dsl.execute(steps) == pytest.approx(825 / 95)


def test_execute_greater_and_exp():
    assert dsl.execute(dsl.parse_program("greater(5, 3)")) == "yes"
    assert dsl.execute(dsl.parse_program("greater(2, 3)")) == "no"
    assert dsl.execute(dsl.parse_program("exp(2, const_3)")) == 8.0


def test_execute_table_ops_last_duplicate_and_percent_cells():
    assert dsl.execute(dsl.parse_program("table_sum(revenue ( millions ), none)"), TABLE) == 240.0
    # duplicate 'margin' rows: last one wins; cells like '20% ( 20 % )' parse to 0.20
    assert dsl.execute(dsl.parse_program("table_average(margin, none)"), TABLE) == pytest.approx(0.22)
    assert dsl.execute(dsl.parse_program("table_max(margin, none)"), TABLE) == pytest.approx(0.24)


@pytest.mark.parametrize("prog", ["divide(1, 0)", "add(#0, 1)", "add(abc, 1)", "table_sum(nope, none)",
                                  "greater(1, 2), add(#0, 1)"])
def test_execute_errors(prog):
    with pytest.raises(ExecError):
        dsl.execute(dsl.parse_program(prog), TABLE)


def test_try_execute_returns_none_on_failure():
    assert dsl.try_execute("divide(1, 0)") is None
    assert dsl.try_execute("garbage") is None


def test_canonical_commutative_and_independent_steps():
    a = dsl.canonical(dsl.parse_program("add(1, 2), multiply(#0, 3)"))
    b = dsl.canonical(dsl.parse_program("add(2, 1), multiply(3, #0)"))
    assert a == b
    # independent steps in different order
    c = dsl.canonical(dsl.parse_program("subtract(5, 1), subtract(9, 2), divide(#0, #1)"))
    d = dsl.canonical(dsl.parse_program("subtract(9, 2), subtract(5, 1), divide(#1, #0)"))
    assert c == d
    # non-commutative operand order matters
    assert dsl.canonical(dsl.parse_program("subtract(1, 2)")) != dsl.canonical(dsl.parse_program("subtract(2, 1)"))
    assert dsl.canonical(dsl.parse_program("divide(1, 2)")) != dsl.canonical(dsl.parse_program("divide(2, 1)"))


def test_canonical_number_normalisation_and_bad_refs():
    assert dsl.canonical(dsl.parse_program("add(5.0, 2)")) == dsl.canonical(dsl.parse_program("add(2, 5)"))
    with pytest.raises(ProgramError):
        dsl.canonical(dsl.parse_program("add(#1, 2), add(1, 1)"))


def test_skeleton_masks_operands_but_keeps_constants():
    s1 = dsl.skeleton(dsl.parse_program("divide(637, const_100)"))
    s2 = dsl.skeleton(dsl.parse_program("divide(12, const_100)"))
    s3 = dsl.skeleton(dsl.parse_program("divide(12, const_1000)"))
    assert s1 == s2 != s3
    assert dsl.skeleton(dsl.parse_program("table_sum(foo, none)")) == "table_sum(ROW,none)"
