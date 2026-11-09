"""Grounding: turning retrieved facts into copyable *operand candidates* and
mapping gold program arguments onto them (own implementation).

The generator emits a program token by token.  Each token is either
  * a RESERVED token  - operator, punctuation, EOF, constant (const_100), step
                        reference (#0), or 'none'; or
  * a COPIED operand  - a number or a table-row label taken from the input.

Candidates are extracted from the text of the facts that are fed to the
generator.  A gold argument is *grounded* if some candidate carries the same
value (numbers) or the same text (row labels).  If several candidates match
(the same number appears twice) all of them count as correct and the loss
marginalises over them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Example
from .dsl import (ALL_OPS, EOF, Step, TABLE_OPS, program_to_tokens, str_to_num)

# ---- reserved vocabulary -----------------------------------------------------
SOS = "SOS"
PUNCT = ["(", ")", ","]
CONSTS = ["const_1", "const_2", "const_3", "const_4", "const_5", "const_6", "const_7",
          "const_8", "const_9", "const_10", "const_100", "const_1000", "const_10000",
          "const_100000", "const_1000000", "const_1000000000", "const_m1"]
MAX_REFS = 8
REFS = [f"#{i}" for i in range(MAX_REFS)]
RESERVED: List[str] = [SOS, EOF] + PUNCT + list(ALL_OPS) + ["none"] + CONSTS + REFS
RESERVED_IDX: Dict[str, int] = {t: i for i, t in enumerate(RESERVED)}

_NUM_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?|(?<![\w])-?\.\d+%?")


@dataclass
class Candidate:
    kind: str            # "num" | "row"
    text: str            # surface string (as in the document)
    key: object          # numeric value (rounded) for num, lowercase label for row
    fact_id: str
    start: int           # char offsets inside the concatenated fact string
    end: int


def _num_key(s: str) -> Optional[float]:
    v = str_to_num(s)
    return None if v is None else round(v, 8)


def ordered_fact_ids(ex: Example, fact_ids: Sequence[str]) -> List[str]:
    """Deterministic, label-free order: text sentences first, then table rows."""
    def key(fid):
        kind, idx = fid.split("_")
        return (0 if kind == "text" else 1, int(idx))
    return sorted(set(fact_ids), key=key)


def build_input(ex: Example, fact_ids: Sequence[str], sep: str = "</s>"
                ) -> Tuple[str, str, List[Candidate]]:
    """Return (question, facts_text, candidates) for the given facts.

    ``facts_text`` joins the facts with ``sep``; candidates carry char offsets
    into it so they can be aligned to subword tokens.
    """
    parts, cands, pos = [], [], 0
    glue = f" {sep} "
    for fid in ordered_fact_ids(ex, fact_ids):
        text = ex.fact_text(fid)
        base = pos
        if fid.startswith("table_") and int(fid.split("_")[1]) > 0:
            label = str(ex.table[int(fid.split("_")[1])][0]).strip()
            # row label sits right after the leading "the " of the serialised row
            m = re.search(r"\bthe " + re.escape(label) + r" of ", text)
            if label and m:
                s = m.start() + 4
                cands.append(Candidate("row", label, label.lower(), fid, base + s, base + s + len(label)))
        for m in _NUM_RE.finditer(text):
            k = _num_key(m.group(0))
            if k is not None:
                cands.append(Candidate("num", m.group(0), k, fid, base + m.start(), base + m.end()))
        parts.append(text)
        pos += len(text) + len(glue)
    return ex.question, glue.join(parts), cands


# ---- gold program -> targets -------------------------------------------------

def gold_token_targets(steps: Sequence[Step], cands: Sequence[Candidate]
                       ) -> Tuple[List[Tuple[str, object]], bool]:
    """Express the gold program as a target sequence over reserved/candidate slots.

    Returns (targets, grounded).  Each target is ("res", reserved_index) or
    ("cand", [indices of all matching candidates]).  ``grounded`` is False when
    some document operand has no matching candidate (then that target is
    ("miss", None)).
    """
    by_num: Dict[object, List[int]] = {}
    by_row: Dict[str, List[int]] = {}
    for i, c in enumerate(cands):
        (by_num if c.kind == "num" else by_row).setdefault(c.key, []).append(i)

    targets: List[Tuple[str, object]] = []
    grounded = True
    for tok in program_to_tokens(steps):
        if tok in RESERVED_IDX:
            targets.append(("res", RESERVED_IDX[tok]))
        else:
            k = _num_key(tok)
            hits = by_num.get(k) if k is not None else by_row.get(tok.strip().lower())
            if hits:
                targets.append(("cand", list(hits)))
            else:
                targets.append(("miss", None))
                grounded = False
    return targets, grounded


def operand_coverage(ex: Example, fact_ids: Sequence[str]) -> Tuple[int, int]:
    """(# gold document operands grounded, # gold document operands) for given facts."""
    _, _, cands = build_input(ex, fact_ids)
    targets, _ = gold_token_targets(ex.steps, cands)
    doc = [t for t in targets if t[0] in ("cand", "miss")]
    return sum(1 for t in doc if t[0] == "cand"), len(doc)
