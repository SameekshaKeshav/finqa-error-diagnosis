"""Figure: full-evidence recall of the released baseline retriever as a function
of the retrieval budget k, split by gold-evidence modality (test set), plus the
share of all retrieval failures that each modality accounts for.

Usage: python scripts/04_plot_retrieval.py --data data/raw
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.taxonomy import retrieval_breakdown, retrieval_status, gold_axes  # noqa: E402

COL = {"table": "#4C72B0", "text": "#DD8452", "hybrid": "#55A868", "all": "#333333"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--fig", default="report/figures/retrieval_released.pdf")
    ap.add_argument("--out", default="results/retrieval_released_by_k.json")
    a = ap.parse_args()

    exs = load_all(a.data)["test"]
    ranked = {e.id: [f for f, _ in e.released] for e in exs}
    ks = [1, 2, 3, 4, 5]
    curves = {m: [] for m in ("table", "text", "hybrid", "all")}
    for k in ks:
        by_mod = retrieval_breakdown(exs, ranked, "modality", k=k)
        for m in ("table", "text", "hybrid"):
            curves[m].append(by_mod[m]["full_recall"])
        curves["all"].append(retrieval_breakdown(exs, ranked, "all", k=k)["all"]["full_recall"])

    # share of top-5 retrieval failures by modality
    fails = {"table": 0, "text": 0, "hybrid": 0}
    n_by_mod = {"table": 0, "text": 0, "hybrid": 0}
    for e in exs:
        m = gold_axes(e)["modality"]
        n_by_mod[m] += 1
        if retrieval_status(e.gold_ids, ranked[e.id]) != "complete":
            fails[m] += 1
    total_fail = sum(fails.values())
    summary = {"k": ks, "full_recall_by_k": curves, "n_by_modality": n_by_mod,
               "top5_failures_by_modality": fails, "top5_failures_total": total_fail,
               "failure_share": {m: fails[m] / total_fail for m in fails},
               "question_share": {m: n_by_mod[m] / len(exs) for m in n_by_mod}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)
    print(json.dumps({k: summary[k] for k in ("top5_failures_by_modality", "failure_share", "question_share")}, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    for m in ("table", "text", "hybrid", "all"):
        ax[0].plot(ks, curves[m], marker="o", ms=3.5, lw=1.6 if m != "all" else 1.2,
                   ls="-" if m != "all" else "--", color=COL[m], label=m)
        ax[0].text(5.08, curves[m][-1], f"{curves[m][-1]:.2f}", fontsize=7, va="center", color=COL[m])
    ax[0].set_xticks(ks); ax[0].set_xlim(0.8, 5.6); ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("retrieval budget k (facts passed to generator)")
    ax[0].set_ylabel("share with ALL gold facts retrieved")
    ax[0].set_title("(a) Full-evidence recall by modality (test)", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False, loc="lower right")

    mods = ["table", "text", "hybrid"]
    xs = range(len(mods))
    q = [100 * summary["question_share"][m] for m in mods]
    f = [100 * summary["failure_share"][m] for m in mods]
    w = 0.36
    ax[1].bar([x - w / 2 for x in xs], q, w, color="#BBBBBB", label="share of questions")
    ax[1].bar([x + w / 2 for x in xs], f, w, color=[COL[m] for m in mods], label="share of top-5 failures")
    for x, v in zip(xs, q):
        ax[1].text(x - w / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
    for x, v in zip(xs, f):
        ax[1].text(x + w / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels(mods); ax[1].set_ylim(0, 80)
    ax[1].set_ylabel("%"); ax[1].set_title("(b) Where retrieval failures come from", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False, loc="upper right")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig), exist_ok=True)
    fig.savefig(a.fig)
    print("wrote", a.fig)


if __name__ == "__main__":
    main()
