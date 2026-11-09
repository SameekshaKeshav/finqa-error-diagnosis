"""Retrieval-side error analysis using the retriever outputs released with FinQA.

Each FinQA json file ships the baseline retriever's top-5 facts per question.
This script measures how often those facts contain ALL gold evidence, broken
down by the gold-derived taxonomy axes, and compares against random chance.
Needs no GPU and no training; the same functions score my own retriever later.

CAVEAT (stated in the report): the provenance of the released retrieval
results (before/after the upstream May-2022 formatting-bug fix) is not
documented, so treat them as indicative and regenerate with my retriever.

Usage: python scripts/02_retrieval_analysis.py --data data/raw
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.taxonomy import chance_full_recall, retrieval_breakdown  # noqa: E402

AXES = ["modality", "steps_bin", "op_family", "n_gold_bin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results/retrieval_released.json")
    a = ap.parse_args()

    D = load_all(a.data)
    result = {}
    for split in ("dev", "test", "train"):
        exs = D[split]
        ranked = {e.id: [fid for fid, _ in e.released] for e in exs}
        r = {"n": len(exs),
             "overall": retrieval_breakdown(exs, ranked, "all")["all"],
             "chance_full_recall_top5": chance_full_recall(exs, 5),
             "recall_at_k": {str(k): retrieval_breakdown(exs, ranked, "all", k=k)["all"]
                             for k in range(1, 6)},
             "by_axis": {ax: retrieval_breakdown(exs, ranked, ax) for ax in AXES}}
        result[split] = r
        o = r["overall"]
        print(f"\n== {split} (n={len(exs)}) ==  top-5 full recall {o['full_recall']:.3f} | "
              f"fact recall {o['fact_recall']:.3f} | random-chance full recall {r['chance_full_recall_top5']:.3f}")
        for ax in ("modality", "steps_bin", "op_family"):
            for cat, v in r["by_axis"][ax].items():
                print(f"   {ax:10s} {cat:12s} n={v['n']:4d}  full {v['full_recall']:.3f}  fact {v['fact_recall']:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(result, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
