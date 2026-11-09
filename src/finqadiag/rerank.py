"""Execution-guided reranking of beam-search programs (own implementation, no torch).

The generator proposes up to k programs per question (beam search).  Because every
program can be *executed*, candidates can be compared by the answer they produce,
not only by their token probability.  Selection modes:

  top1       highest-log-probability beam (what plain beam search returns)
  exec_top1  highest-log-probability beam that parses and executes
  marginal   answer-level voting: probability mass of all executable beams that
             give the same answer (rounded to 5 decimals); the answer with the most
             mass wins and its most probable program is returned.  ``temp`` flattens
             (temp>1) or sharpens (temp<1) the beam distribution before voting.

No parameter is learned; k, mode and temp are chosen on the dev set.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Example
from .dsl import ProgramError, Step, parse_tokens, try_execute
from .metrics import answers_match, execution_correct, program_correct

Beams = List[Dict[str, object]]          # [{"tokens": [...], "logp": float}, ...] best first


def _parse(tokens) -> Optional[List[Step]]:
    try:
        return parse_tokens(tokens)
    except ProgramError:
        return None


def _key(v):
    return v if isinstance(v, str) else round(float(v), 5)


def executed(beams: Beams, ex: Example) -> List[Tuple[Optional[List[Step]], object, float]]:
    """(steps, answer-or-None, logp) per beam."""
    out = []
    for b in beams:
        steps = _parse(b["tokens"])
        val = try_execute(list(steps), ex.table) if steps else None
        out.append((steps, val, float(b["logp"])))
    return out


def select(beams: Beams, ex: Example, mode: str = "marginal", temp: float = 1.0,
           k: Optional[int] = None) -> Optional[List[Step]]:
    beams = beams[:k] if k else beams
    if not beams:
        return None
    ex_all = executed(beams, ex)
    if mode == "top1":
        return ex_all[0][0]
    ok = [(s, v, lp) for s, v, lp in ex_all if s is not None and v is not None]
    if not ok:
        return ex_all[0][0]
    if mode == "exec_top1":
        return ok[0][0]
    if mode == "marginal":
        mx = max(lp for _, _, lp in ok)
        mass: Dict[object, float] = defaultdict(float)
        best: Dict[object, Tuple[float, List[Step]]] = {}
        for s, v, lp in ok:
            key = _key(v)
            mass[key] += math.exp((lp - mx) / temp)
            if key not in best or lp > best[key][0]:
                best[key] = (lp, s)
        win = max(mass, key=lambda x: mass[x])
        return best[win][1]
    raise ValueError(mode)


def evaluate(preds: Dict[str, Dict[str, object]], examples: Sequence[Example], mode: str,
             temp: float = 1.0, k: Optional[int] = None) -> Tuple[float, float]:
    ea = pa = 0
    for ex in examples:
        steps = select(preds[ex.id]["beams"], ex, mode, temp, k)
        ea += execution_correct(steps, ex.table, ex.exe_ans)
        pa += program_correct(steps, ex.program)
    n = len(examples)
    return ea / n, pa / n


def oracle_at_k(preds: Dict[str, Dict[str, object]], examples: Sequence[Example], k: int) -> float:
    """Share of questions where at least one of the first k beams gives the gold answer."""
    hit = 0
    for ex in examples:
        for s, v, _ in executed(preds[ex.id]["beams"][:k], ex):
            if v is not None and answers_match(v, ex.exe_ans):
                hit += 1
                break
    return hit / len(examples)


def per_example_correct(preds, examples, mode, temp=1.0, k=None) -> Dict[str, bool]:
    return {ex.id: bool(execution_correct(select(preds[ex.id]["beams"], ex, mode, temp, k),
                                          ex.table, ex.exe_ans)) for ex in examples}
