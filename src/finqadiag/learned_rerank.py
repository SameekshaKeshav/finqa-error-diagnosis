"""Learned reranker over beam candidates (own implementation; scikit-learn model, no torch).

Each beam candidate becomes one feature row; the label is "its executed answer is
correct".  A classifier is trained on the DEV beams (the generator never saw dev
questions in training) and applied to TEST: per question the executable candidate
with the highest predicted probability is returned.  Features are cheap and
interpretable: model score and rank, agreement with the other beams (answer vote
share), program shape, answer plausibility, and a few question cues.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import Example
from .metrics import answers_match, execution_correct
from .rerank import _key, executed

OPS = ["add", "subtract", "multiply", "divide", "exp", "greater", "table_max", "table_min",
       "table_sum", "table_average"]
CUES = ["percent", "ratio", "average|mean", "total|sum", "change|difference|increase|decrease",
        "how much|what was|what is", "portion|fraction|proportion", "growth|rate"]
FEATURES: List[str] = (
    ["logp", "logp_gap", "rank", "is_top", "n_steps", "n_const", "n_tokens", "vote_share", "vote_count",
     "n_answers", "top_mass", "is_int", "log_mag", "negative", "in_unit", "in_pct", "pct_cue_unit",
     "pct_cue_x100", "len_vs_top"]
    + [f"last_{o}" for o in OPS] + [f"cue{i}" for i in range(len(CUES))])


def build_rows(beams: Sequence[dict], ex: Example, temp: float = 1.0
               ) -> Tuple[np.ndarray, List[int], List[Optional[list]]]:
    """Feature matrix over the executable candidates of one question.

    Returns (X, beam_index_of_each_row, steps_of_each_row).
    """
    cands = executed(list(beams), ex)
    ok = [(i, s, v, lp) for i, (s, v, lp) in enumerate(cands) if s is not None and v is not None]
    if not ok:
        return np.zeros((0, len(FEATURES))), [], []
    best = cands[0][2]
    mx = max(lp for _, _, _, lp in ok)
    mass: Dict[object, float] = defaultdict(float)
    cnt: Dict[object, int] = defaultdict(int)
    for _, _, v, lp in ok:
        mass[_key(v)] += math.exp((lp - mx) / temp)
        cnt[_key(v)] += 1
    tot = sum(mass.values())
    top_len = len(cands[0][0]) if cands[0][0] else 0
    q = ex.question.lower()
    cues = [1.0 if re.search(c, q) else 0.0 for c in CUES]
    rows, idxs, steps = [], [], []
    for i, s, v, lp in ok:
        isnum = not isinstance(v, str)
        val = float(v) if isnum else 0.0
        last = s[-1].op
        const_n = sum(a.startswith("const_") for st in s for a in st.args)
        pct_cue = cues[0]
        x = [lp, best - lp, i, float(i == 0), len(s), const_n, sum(1 + len(st.args) for st in s),
             mass[_key(v)] / tot, cnt[_key(v)] / len(ok), len(mass), max(mass.values()) / tot,
             float(isnum and abs(val - round(val)) < 1e-9), math.log10(abs(val) + 1e-9) if isnum else 0.0,
             float(isnum and val < 0), float(isnum and 0 <= val <= 1), float(isnum and 0 <= val <= 100),
             pct_cue * float(isnum and 0 <= val <= 1),
             pct_cue * float(last == "multiply" and any(a == "const_100" for a in s[-1].args)),
             len(s) - top_len]
        x += [float(last == o) for o in OPS] + cues
        rows.append(x); idxs.append(i); steps.append(s)
    return np.array(rows, dtype=float), idxs, steps


def build_dataset(preds: Dict[str, dict], examples: Sequence[Example], temp: float = 1.0):
    """Rows for all questions: X, y (execution-correct), group (question index), steps, answer keys, logp."""
    X, y, g, S, K, L = [], [], [], [], [], []
    for qi, ex in enumerate(examples):
        beams = preds[ex.id]["beams"]
        Xi, idxs, steps = build_rows(beams, ex, temp)
        cands = executed(list(beams), ex)
        for r, i, s in zip(Xi, idxs, steps):
            X.append(r); g.append(qi); S.append(s)
            K.append(_key(cands[i][1])); L.append(cands[i][2])
            y.append(float(execution_correct(s, ex.table, ex.exe_ans)))
    return np.array(X), np.array(y), np.array(g), S, K, np.array(L)


def pick(prob: np.ndarray, groups: np.ndarray, steps: list, keys: list, logp: np.ndarray,
         fallback: list) -> list:
    """Per question: the ANSWER whose best candidate has the highest predicted probability wins;
    return that answer's most probable (highest log-probability) program.  The classifier decides
    the answer, the generator decides the program, so program accuracy is not degraded when
    several programs give the same answer.  Falls back to the top-1 beam when nothing executes."""
    out = list(fallback)
    ans: Dict[Tuple[int, object], float] = {}
    prog: Dict[Tuple[int, object], Tuple[float, list]] = {}
    for p, gi, s, k, lp in zip(prob, groups, steps, keys, logp):
        ans[(gi, k)] = max(ans.get((gi, k), -1.0), p)
        if (gi, k) not in prog or lp > prog[(gi, k)][0]:
            prog[(gi, k)] = (lp, s)
    best: Dict[int, Tuple[float, object]] = {}
    for (gi, k), p in ans.items():
        if gi not in best or p > best[gi][0]:
            best[gi] = (p, k)
    for gi, (_, k) in best.items():
        out[gi] = prog[(gi, k)][1]
    return out
