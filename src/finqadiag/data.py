"""Dataset loading and preprocessing for FinQA (own implementation).

Facts
-----
A document is a bag of *facts* that evidence can come from:
  * ``text_i``  - the i-th sentence of pre_text + post_text (concatenated), and
  * ``table_j`` - the j-th table row (row 0 is the header row).
``qa.gold_inds`` names the gold supporting facts of a question.

Row serialisation (``row_to_text``) turns a table row into a sentence.  It
depends ONLY on the table contents and the row index, never on gold
annotations, so positive and negative retriever examples are formatted
identically (the upstream project once shipped a bug where inconsistent
formatting leaked the label to the retriever).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .dsl import Step, parse_program
from .metrics import gold_answer


@dataclass
class Example:
    id: str
    question: str
    pre_text: List[str]
    post_text: List[str]
    table: List[List[str]]
    program: str
    steps: List[Step]
    exe_ans: object
    gold_ids: List[str]                       # e.g. ["table_3", "text_12"]
    released: Optional[List[Tuple[str, float]]] = None  # authors' retriever output, best first

    # ---- facts -----------------------------------------------------------
    @property
    def texts(self) -> List[str]:
        return list(self.pre_text) + list(self.post_text)

    def fact_ids(self) -> List[str]:
        return [f"text_{i}" for i in range(len(self.texts))] + \
               [f"table_{j}" for j in range(len(self.table))]

    def fact_text(self, fid: str) -> str:
        kind, idx = fid.split("_")
        idx = int(idx)
        return self.texts[idx] if kind == "text" else row_to_text(self.table, idx)

    def facts(self) -> Dict[str, str]:
        return {fid: self.fact_text(fid) for fid in self.fact_ids()}


def row_to_text(table: Sequence[Sequence[str]], j: int) -> str:
    """Serialise table row ``j`` as a sentence.

    Header row (j=0): its cells joined by ' ; '.
    Other rows: "<header0> the <row label> of <col header> is <cell> ; ..."
    """
    header, row = table[0], table[j]
    if j == 0:
        return " ; ".join(str(c) for c in row if str(c).strip())
    prefix = f"{header[0]} " if header and str(header[0]).strip() else ""
    label = row[0]
    parts = [f"the {label} of {header[k]} is {row[k]} ;"
             for k in range(1, min(len(row), len(header)))]
    return (prefix + " ".join(parts)).strip()


def _released_list(raw: dict) -> Optional[List[Tuple[str, float]]]:
    """Merge the authors' released table/text retrieval results into one ranked list."""
    items = []
    for key in ("table_retrieved", "text_retrieved"):
        for r in raw.get(key) or []:
            items.append((r["ind"], float(r["score"])))
    if not items:
        return None
    return sorted(items, key=lambda t: -t[1])


def load_split(path: str) -> List[Example]:
    with open(path) as f:
        raw = json.load(f)
    out = []
    for r in raw:
        q = r["qa"]
        n_text = len(r["pre_text"]) + len(r["post_text"])
        n_rows = len(r["table"])
        # Drop gold ids that do not point at a real fact (one such annotation
        # artefact exists: train RE/2010/page_120.pdf-2 has 'text_-1').
        gold = [g for g in q["gold_inds"].keys()
                if 0 <= _fact_sort_key(g)[1] < (n_text if g.startswith("text") else n_rows)]
        out.append(Example(
            id=r["id"], question=q["question"],
            pre_text=r["pre_text"], post_text=r["post_text"], table=r["table"],
            program=q["program"], steps=parse_program(q["program"]),
            exe_ans=gold_answer(q["exe_ans"]),
            gold_ids=sorted(gold, key=_fact_sort_key),
            released=_released_list(r),
        ))
    return out


def _fact_sort_key(fid: str):
    kind, idx = fid.split("_")
    return (kind, int(idx))


def load_all(data_dir: str = "data/raw") -> Dict[str, List[Example]]:
    return {s: load_split(os.path.join(data_dir, f"{s}.json")) for s in ("train", "dev", "test")}
