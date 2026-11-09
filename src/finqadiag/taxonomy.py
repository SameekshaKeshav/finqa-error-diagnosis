"""Error taxonomy for retriever-generator numerical QA (core contribution).

Every test example is tagged along axes that answer *where* and *how* a model
fails, not just how often.

Gold-derived axes (known without any model):
  steps_bin     1 / 2 / 3+ program steps
  complexity    single-step vs multi-step
  modality      table | text | hybrid  (which kinds of facts the gold evidence uses)
  op_family     table_agg > comparison > ratio > add_sub_mul  (first that applies)
  n_gold_bin    1 / 2 / 3+ gold supporting facts

Prediction-derived axes:
  retrieval     complete | partial | miss   (are the gold facts inside the retrieved set?)
  correct_ea    predicted program executes to the gold answer
  correct_pa    predicted program is symbolically identical to gold
  error_type    non_executable | structure_error | operand_error  (for wrong answers)
                structure_error: different operators/constants (skeleton differs)
                operand_error:   same skeleton, wrong document operand(s)
  locus         generation | retrieval | both  (for wrong answers; needs an oracle run)
                = generation if all gold evidence was retrieved;
                  otherwise retrieval if the generator succeeds when handed the gold
                  evidence, and both if it fails even then.  Without an oracle run,
                  incomplete retrieval is reported as 'retrieval_or_both'.
  rel_err_bin   magnitude of the numeric error for executable wrong answers.

All of this is original code.  The locus definition (an oracle-evidence
counterfactual) is a design choice of this project.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from .data import Example
from .dsl import (ExecError, ProgramError, Step, TABLE_OPS, parse_tokens, skeleton, try_execute)
from .grounding import operand_coverage
from .metrics import answers_match, execution_correct, program_correct


# ---- gold-derived axes -----------------------------------------------------

def op_family(steps: Sequence[Step]) -> str:
    ops = {s.op for s in steps}
    if ops & set(TABLE_OPS):
        return "table_agg"
    if "greater" in ops:
        return "comparison"
    if "divide" in ops:
        return "ratio"
    return "add_sub_mul"


def _bin3(n: int) -> str:
    return str(n) if n < 3 else "3+"


def modality(gold_ids: Iterable[str]) -> str:
    kinds = {g.split("_")[0] for g in gold_ids}
    if kinds == {"table"}:
        return "table"
    if kinds == {"text"}:
        return "text"
    return "hybrid" if kinds else "none"


def gold_axes(ex: Example) -> Dict[str, str]:
    n = len(ex.steps)
    return {
        "steps_bin": _bin3(n),
        "complexity": "single-step" if n == 1 else "multi-step",
        "modality": modality(ex.gold_ids),
        "op_family": op_family(ex.steps),
        "n_gold_bin": _bin3(len(ex.gold_ids)),
    }


# ---- prediction-derived axes ----------------------------------------------

def retrieval_status(gold_ids: Sequence[str], retrieved_ids: Iterable[str]) -> str:
    got = set(retrieved_ids)
    hit = sum(1 for g in gold_ids if g in got)
    if hit == len(gold_ids):
        return "complete"
    return "partial" if hit else "miss"


def rel_err_bin(pred_val, gold_val) -> str:
    if isinstance(pred_val, str) or isinstance(gold_val, str):
        return "yes/no flip"
    denom = abs(gold_val) if gold_val else 1.0
    r = abs(pred_val - gold_val) / denom
    return "<1%" if r < 0.01 else "1-10%" if r < 0.10 else ">10%"


def error_type(pred_steps: Optional[Sequence[Step]], ex: Example) -> str:
    """Classify a *wrong* prediction."""
    if pred_steps is None or try_execute(list(pred_steps), ex.table) is None:
        return "non_executable"
    try:
        same_skeleton = skeleton(list(pred_steps)) == skeleton(ex.steps)
    except ProgramError:
        return "non_executable"
    return "operand_error" if same_skeleton else "structure_error"


def tag_example(ex: Example,
                pred_steps: Optional[Sequence[Step]],
                retrieved_ids: Iterable[str],
                oracle_pred_steps: Optional[Sequence[Step]] = None,
                have_oracle: bool = False) -> Dict[str, object]:
    """Tag one example. ``pred_steps`` is None if the model's output was malformed."""
    tag: Dict[str, object] = {"id": ex.id, **gold_axes(ex)}
    retrieved_ids = list(retrieved_ids)
    ret = retrieval_status(ex.gold_ids, retrieved_ids)
    ea = execution_correct(pred_steps, ex.table, ex.exe_ans)
    pa = program_correct(pred_steps, ex.program)
    got, need = operand_coverage(ex, retrieved_ids)
    # operands_grounded: every gold operand is copyable from the retrieved text
    # (can hold even when a gold *fact* was missed, if its numbers recur elsewhere)
    tag.update(retrieval=ret, correct_ea=ea, correct_pa=pa, operands_grounded=(got == need))
    if ea:
        tag.update(error_type="correct", locus="correct", rel_err_bin="")
        return tag

    tag["error_type"] = error_type(pred_steps, ex)
    val = try_execute(list(pred_steps), ex.table) if pred_steps else None
    tag["rel_err_bin"] = rel_err_bin(val, ex.exe_ans) if val is not None else ""
    if ret == "complete":
        tag["locus"] = "generation"
    elif have_oracle:
        oracle_ok = execution_correct(oracle_pred_steps, ex.table, ex.exe_ans)
        tag["locus"] = "retrieval" if oracle_ok else "both"
    else:
        tag["locus"] = "retrieval_or_both"
    return tag


def _steps_from_tokens(tokens) -> Optional[List[Step]]:
    try:
        return parse_tokens(tokens)
    except ProgramError:
        return None


def tags_from_predictions(examples: Sequence[Example],
                          preds: Dict[str, Dict[str, object]],
                          oracle_preds: Optional[Dict[str, Dict[str, object]]] = None
                          ) -> List[Dict[str, object]]:
    """Tag every example from a predictions file.

    ``preds[id] = {"tokens": [...program tokens...], "facts": [fact ids fed to the generator]}``.
    ``oracle_preds`` has the same layout, produced with oracle_selector inputs; when
    given, retrieval-incomplete failures are split into retrieval vs both.
    """
    tags = []
    for ex in examples:
        p = preds[ex.id]
        o = oracle_preds[ex.id] if oracle_preds else None
        tags.append(tag_example(
            ex, _steps_from_tokens(p["tokens"]), p["facts"],
            oracle_pred_steps=_steps_from_tokens(o["tokens"]) if o else None,
            have_oracle=o is not None))
    return tags


def locus_totals(tags: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """Counts of correct / generation / retrieval / both (/ retrieval_or_both)."""
    return dict(Counter(str(t["locus"]) for t in tags))


# ---- retrieval-only analysis (needs no generator) -----------------------------

def retrieval_breakdown(examples: Sequence[Example],
                        ranked: Dict[str, Sequence[str]],
                        axis: str,
                        k: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Retrieval quality per category of a gold-derived axis.

    ``ranked[id]`` lists fact ids best-first.  With ``k`` only the top-k count.
      full_recall: share of questions whose gold facts are ALL retrieved
      fact_recall: share of individual gold facts retrieved
    """
    agg: Dict[str, List[float]] = defaultdict(lambda: [0, 0, 0, 0])  # n, full, hits, golds
    for ex in examples:
        got = list(ranked[ex.id])[:k] if k else list(ranked[ex.id])
        hit = sum(1 for g in ex.gold_ids if g in set(got))
        a = agg[gold_axes(ex)[axis] if axis != "all" else "all"]
        a[0] += 1
        a[1] += hit == len(ex.gold_ids)
        a[2] += hit
        a[3] += len(ex.gold_ids)
    return {c: {"n": a[0], "full_recall": a[1] / a[0], "fact_recall": a[2] / a[3]}
            for c, a in sorted(agg.items())}


def chance_full_recall(examples: Sequence[Example], k: int) -> float:
    """Expected full-recall of picking k facts uniformly at random (hypergeometric)."""
    from math import comb
    tot = 0.0
    for ex in examples:
        n, g = len(ex.fact_ids()), len(ex.gold_ids)
        tot += comb(n - g, k - g) / comb(n, k) if k >= g else 0.0
    return tot / len(examples)


# ---- aggregation -------------------------------------------------------------

def breakdown(tags: List[Dict[str, object]], axis: str) -> Dict[str, Dict[str, object]]:
    """Per-category accuracy and error composition along one axis."""
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for t in tags:
        groups[str(t[axis])].append(t)
    out = {}
    for k, ts in sorted(groups.items()):
        n = len(ts)
        wrong = [t for t in ts if not t["correct_ea"]]
        out[k] = {
            "n": n,
            "ea": sum(bool(t["correct_ea"]) for t in ts) / n,
            "pa": sum(bool(t["correct_pa"]) for t in ts) / n,
            "retrieval_complete": sum(t["retrieval"] == "complete" for t in ts) / n,
            "n_wrong": len(wrong),
            "locus": dict(Counter(t["locus"] for t in wrong)),
            "error_type": dict(Counter(t["error_type"] for t in wrong)),
        }
    return out
