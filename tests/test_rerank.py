import math

from conftest import make_example
from finqadiag.dsl import parse_program, program_to_tokens
from finqadiag.rerank import evaluate, oracle_at_k, select


def beam(prog, logp):
    return {"tokens": program_to_tokens(parse_program(prog)), "logp": logp}


GOOD = "divide(637, const_5)"          # 127.4  (gold)
BAD_A = "divide(637, 5.0)"             # 127.4 too (same answer, different program)
BAD_B = "multiply(637, 5.0)"           # 3185
BAD_C = "subtract(637, 5.0)"           # 632
EX = make_example()


def test_top1_takes_best_logp_even_if_wrong():
    b = [beam(BAD_B, -0.1), beam(GOOD, -0.5)]
    assert select(b, EX, "top1")[0].op == "multiply"


def test_exec_top1_skips_unparsable():
    b = [{"tokens": ["divide", "(", "637"], "logp": -0.01}, beam(GOOD, -2.0)]
    assert select(b, EX, "exec_top1")[0].op == "divide"


def test_marginal_votes_over_programs_with_same_answer():
    # two different programs agree on 127.4, together they outweigh one slightly likelier wrong beam
    b = [beam(BAD_B, -1.0), beam(GOOD, -1.2), beam(BAD_A, -1.3)]
    steps = select(b, EX, "marginal", 1.0)
    assert steps[0].op == "divide"
    assert steps[0].args == ("637", "const_5")      # most probable program of the winning answer


def test_marginal_temperature_sharpening_recovers_top1():
    b = [beam(BAD_B, -1.0), beam(GOOD, -1.2), beam(BAD_A, -1.3)]
    assert select(b, EX, "marginal", 0.05)[0].op == "multiply"


def test_k_truncation_and_empty():
    b = [beam(BAD_B, -0.1), beam(GOOD, -0.5)]
    assert select(b, EX, "top1", k=1)[0].op == "multiply"
    assert select([], EX, "marginal") is None


def test_evaluate_and_oracle():
    preds = {EX.id: {"beams": [beam(BAD_B, -0.1), beam(BAD_C, -0.2), beam(GOOD, -0.9)]}}
    assert evaluate(preds, [EX], "top1", k=1) == (0.0, 0.0)
    assert oracle_at_k(preds, [EX], 2) == 0.0
    assert oracle_at_k(preds, [EX], 3) == 1.0
    ea, pa = evaluate(preds, [EX], "marginal", 100.0)      # nearly uniform: all answers tie -> best logp
    assert ea in (0.0, 1.0)
