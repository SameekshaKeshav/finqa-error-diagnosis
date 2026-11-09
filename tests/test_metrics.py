import pytest

from finqadiag import dsl, metrics


def steps(p):
    return dsl.parse_program(p)


def test_answers_match_rounding_and_halves():
    assert metrics.answers_match(1.640625, 1.64062)          # exact half rounds like the dataset
    assert metrics.answers_match(127.4, 127.40000001)
    assert not metrics.answers_match(0.00867, 0.00868)
    assert not metrics.answers_match(None, 1.0)
    assert not metrics.answers_match(float("inf"), 1.0)


def test_yes_no():
    assert metrics.answers_match("yes", "yes")
    assert not metrics.answers_match("yes", "no")
    assert not metrics.answers_match(1.0, "yes")


def test_gold_answer_parsing():
    assert metrics.gold_answer("1,234.5") == 1234.5
    assert metrics.gold_answer("Yes") == "yes"
    assert metrics.gold_answer(3) == 3.0


def test_execution_vs_program_correct():
    gold = "add(2, 3), multiply(#0, 4)"
    reordered = steps("add(3, 2), multiply(4, #0)")
    assert metrics.program_correct(reordered, gold)
    assert metrics.execution_correct(reordered, (), 20.0)
    # right answer, different program: EA yes, PA no
    other = steps("multiply(5, 4)")
    assert metrics.execution_correct(other, (), 20.0)
    assert not metrics.program_correct(other, gold)


def test_none_prediction_is_wrong():
    assert not metrics.execution_correct(None, (), 1.0)
    assert not metrics.program_correct(None, "add(1, 2)")
