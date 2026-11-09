"""Why is a wrong program wrong?  Mechanism of the first structural difference (pure Python).

Categories, checked in this order, for a wrong predicted program vs the gold program:
  unparsable          the token sequence is not a valid program
  too_few_steps / too_many_steps   different number of steps
  wrong_operator      same number of steps, some step uses a different operator
  args_swapped        same operators, same arguments per step but in a different order
  wrong_constant      same operators; differs only where a constant is involved
  wrong_reference     same operators; a step reference (#k) differs
  wrong_operand       same operators and structure; a document number / row label differs
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .dsl import Step

CATEGORIES = ["unparsable", "too_few_steps", "too_many_steps", "wrong_operator",
              "args_swapped", "wrong_constant", "wrong_reference", "wrong_operand"]


def mechanism(pred: Optional[Sequence[Step]], gold: Sequence[Step]) -> str:
    if pred is None:
        return "unparsable"
    if len(pred) != len(gold):
        return "too_few_steps" if len(pred) < len(gold) else "too_many_steps"
    if [s.op for s in pred] != [s.op for s in gold]:
        return "wrong_operator"
    pairs = [(a, b) for s, t in zip(pred, gold) for a, b in zip(s.args, t.args)]
    if all(sorted(s.args) == sorted(t.args) for s, t in zip(pred, gold)):
        return "args_swapped"
    if any((a.startswith("const_") or b.startswith("const_")) and a != b for a, b in pairs):
        return "wrong_constant"
    if any(a.startswith("#") != b.startswith("#") or (a.startswith("#") and a != b) for a, b in pairs):
        return "wrong_reference"
    return "wrong_operand"
