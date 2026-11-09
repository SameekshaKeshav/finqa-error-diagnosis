"""Build the error taxonomy from prediction files and plot it.

  python scripts/07_taxonomy_report.py --split test \
      --pred runs/gen_released/pred_test.json \
      --oracle runs/gen_released/pred_test_oracle.json --name baseline

Writes  results/taxonomy_<name>.csv (one row per question),
        results/taxonomy_<name>.json (breakdown tables),
        report/figures/taxonomy_<name>.pdf
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.taxonomy import breakdown, locus_totals, tags_from_predictions  # noqa: E402

AXES = ["modality", "steps_bin", "op_family", "n_gold_bin", "retrieval"]
LOCI = ["correct", "generation", "retrieval", "both", "retrieval_or_both"]
LOCUS_COL = {"correct": "#8FBF8F", "generation": "#C44E52", "retrieval": "#4C72B0",
             "both": "#8172B3", "retrieval_or_both": "#8C8CC0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--name", default="baseline")
    a = ap.parse_args()

    exs = load_all(a.data)[a.split]
    preds = json.load(open(a.pred))
    oracle = json.load(open(a.oracle)) if a.oracle else None
    tags = tags_from_predictions(exs, preds, oracle)

    os.makedirs("results", exist_ok=True)
    with open(f"results/taxonomy_{a.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tags[0].keys()))
        w.writeheader()
        w.writerows(tags)
    n = len(tags)
    summary = {"n": n,
               "ea": sum(bool(t["correct_ea"]) for t in tags) / n,
               "pa": sum(bool(t["correct_pa"]) for t in tags) / n,
               "locus": locus_totals(tags),
               "by_axis": {ax: breakdown(tags, ax) for ax in AXES}}
    json.dump(summary, open(f"results/taxonomy_{a.name}.json", "w"), indent=2)
    print(f"n={n}  EA={summary['ea']:.4f}  PA={summary['pa']:.4f}  locus={summary['locus']}")
    for ax in ("modality", "steps_bin", "op_family"):
        for cat, v in summary["by_axis"][ax].items():
            print(f"  {ax:10s} {cat:12s} n={v['n']:4d} EA={v['ea']:.3f} PA={v['pa']:.3f} "
                  f"retr-complete={v['retrieval_complete']:.3f} errors={v['error_type']}")

    # ---- figure: (a) EA by axis category, (b) outcome composition by modality ----
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1.1, 1.2]})
    labels, vals, ns = [], [], []
    for axname in ("modality", "steps_bin"):
        for cat, v in summary["by_axis"][axname].items():
            labels.append(f"{axname.replace('steps_bin', 'steps')}={cat}")
            vals.append(v["ea"]); ns.append(v["n"])
    ax[0].barh(labels[::-1], vals[::-1], color="#4C72B0")
    for y, (v, k) in enumerate(zip(vals[::-1], ns[::-1])):
        ax[0].text(v + 0.01, y, f"{v:.2f} (n={k})", va="center", fontsize=7)
    ax[0].set_xlim(0, 1.0); ax[0].set_xlabel("execution accuracy")
    ax[0].set_title("(a) EA by gold-derived category", fontsize=9)

    mods = list(summary["by_axis"]["modality"].keys())
    left = [0.0] * len(mods)
    for locus in LOCI:
        share = [summary["by_axis"]["modality"][m]["n"] and
                 sum(1 for t in tags if t["modality"] == m and t["locus"] == locus) /
                 summary["by_axis"]["modality"][m]["n"] for m in mods]
        if not any(share):
            continue
        ax[1].barh(mods, share, left=left, color=LOCUS_COL[locus], label=locus.replace("_", " "))
        left = [l + s for l, s in zip(left, share)]
    ax[1].set_xlim(0, 1.0); ax[1].set_xlabel("share of questions")
    ax[1].set_title("(b) Outcome / failure locus by modality", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs("report/figures", exist_ok=True)
    fig.savefig(f"report/figures/taxonomy_{a.name}.pdf", bbox_inches="tight")
    print("wrote results/ and report/figures/ for", a.name)


if __name__ == "__main__":
    main()
