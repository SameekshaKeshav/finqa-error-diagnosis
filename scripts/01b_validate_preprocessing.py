"""Validate the preprocessing against the dataset's own annotations.

1. Row serialisation: compare data.row_to_text with the text the dataset stores
   for gold table rows (qa.gold_inds).  Reports exact / whitespace-only /
   header-row / other differences.  'other' must be 0 for the serialisation to
   be content-faithful.
2. Operand coverage: share of gold document operands (and of whole programs)
   that are copyable from (a) the full document, (b) the gold facts,
   (c) the released retriever's top-5.  This bounds what any pointer-style
   generator can reach.
3. Annotation artefacts: gold ids that point at non-existent facts.

Usage: python scripts/01b_validate_preprocessing.py --data data/raw
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_split, row_to_text  # noqa: E402
from finqadiag.grounding import operand_coverage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results/preprocessing_validation.json")
    a = ap.parse_args()

    ser, bad_gold = Counter(), []
    cov = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(a.data, f"{split}.json")
        raw = json.load(open(path))
        for r in raw:
            table = r["table"]
            n_text = len(r["pre_text"]) + len(r["post_text"])
            for fid, gtext in r["qa"]["gold_inds"].items():
                kind, i = fid.split("_")
                i = int(i)
                if not 0 <= i < (n_text if kind == "text" else len(table)):
                    bad_gold.append({"split": split, "id": r["id"], "gold_id": fid})
                    continue
                if kind != "table":
                    continue
                mine = row_to_text(table, i)
                if mine == gtext:
                    ser["exact"] += 1
                elif re.sub(r"\s+", "", mine) == re.sub(r"\s+", "", gtext):
                    ser["whitespace_only"] += 1
                elif i == 0:
                    ser["header_row_differs"] += 1
                else:
                    ser["other"] += 1
        exs = load_split(path)
        sels = {"all_facts": lambda e: e.fact_ids(), "gold_facts": lambda e: e.gold_ids,
                "released_top5": lambda e: [f for f, _ in e.released]}
        cov[split] = {}
        for name, sel in sels.items():
            got = need = full = 0
            for e in exs:
                g, n = operand_coverage(e, sel(e))
                got += g; need += n; full += g == n
            cov[split][name] = {"operand_coverage": got / need, "programs_fully_grounded": full / len(exs)}
    out = {"row_serialisation_vs_dataset": dict(ser),
           "row_serialisation_total": sum(ser.values()),
           "gold_ids_pointing_at_nonexistent_facts": bad_gold,
           "operand_coverage": cov}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("row_serialisation_vs_dataset", "row_serialisation_total",
                                         "gold_ids_pointing_at_nonexistent_facts")}, indent=1))
    for s, v in cov.items():
        print(s, {k: (round(x["operand_coverage"], 4), round(x["programs_fully_grounded"], 4)) for k, x in v.items()})


if __name__ == "__main__":
    main()
