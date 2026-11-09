"""Evaluation metrics: execution accuracy (EA) and program accuracy (PA).

* EA  - the predicted program executes and its answer equals the gold answer.
* PA  - the predicted program is symbolically identical to the gold program,
        ignoring operand order of commutative ops and order of independent
        steps (see dsl.canonical).  Operands are compared by VALUE, so a
        reserved constant (const_5) and an equal number copied from the text
        (5) are the same operand; this is slightly more lenient than a strict
        token match and is a deliberate, documented choice.

Written from the metric definitions in the FinQA paper; edge-case handling
(tolerance, yes/no answers) is my own and documented here so results can be
interpreted precisely.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

from .dsl import ProgramError, Step, canonical, parse_program, try_execute, Value

# Gold answers in the dataset are stored rounded to 5 decimals, so predictions
# are compared after the same rounding (a pure tolerance would mis-handle exact
# halves such as 1.640625 -> 1.64062).
DECIMALS = 5


def answers_match(pred: Optional[Value], gold: Value) -> bool:
    if pred is None:
        return False
    if isinstance(gold, str) or isinstance(pred, str):
        return str(pred).strip().lower() == str(gold).strip().lower()
    p, g = float(pred), float(gold)
    if not (math.isfinite(p) and math.isfinite(g)):
        return False
    return round(p, DECIMALS) == round(g, DECIMALS)


def gold_answer(raw) -> Value:
    """Normalise the dataset's exe_ans field (float, numeric string, 'yes'/'no')."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if s.lower() in ("yes", "no"):
        return s.lower()
    return float(s.replace(",", ""))


def execution_correct(pred_steps: Optional[Sequence[Step]], table, gold_exe_ans) -> bool:
    if pred_steps is None:
        return False
    return answers_match(try_execute(list(pred_steps), table), gold_answer(gold_exe_ans))


def program_correct(pred_steps: Optional[Sequence[Step]], gold_program: str) -> bool:
    if pred_steps is None:
        return False
    try:
        return canonical(list(pred_steps)) == canonical(parse_program(gold_program))
    except ProgramError:
        return False
