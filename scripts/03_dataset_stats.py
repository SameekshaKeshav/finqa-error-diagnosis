"""Dataset statistics for the report: sizes, program complexity, evidence
modality, and the complexity x modality cross-tab that forms the denominator of
every taxonomy table.  Also writes the dataset figure.

Usage: python scripts/03_dataset_stats.py --data data/raw
"""
import argparse
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.taxonomy import gold_axes  # noqa: E402

MOD_ORDER = ["table", "text", "hybrid"]
STEP_ORDER = ["1", "2", "3+"]
COL = {"table": "#4C72B0", "text": "#DD8452", "hybrid": "#55A868"}  # colour-blind-safe trio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results/dataset_stats.json")
    ap.add_argument("--fig", default="report/figures/dataset_axes.pdf")
    a = ap.parse_args()

    D = load_all(a.data)
    stats = {}
    for split, exs in D.items():
        axes = [gold_axes(e) for e in exs]
        ops = Counter(s.op for e in exs for s in e.steps)
        stats[split] = {
            "n_questions": len(exs),
            "n_documents_text_sentences_mean": sum(len(e.texts) for e in exs) / len(exs),
            "n_table_rows_mean": sum(len(e.table) for e in exs) / len(exs),
            "n_facts_per_doc_mean": sum(len(e.fact_ids()) for e in exs) / len(exs),
            "tokens_per_doc_mean": sum(len(" ".join(e.facts().values()).split()) for e in exs) / len(exs),
            "n_gold_facts_mean": sum(len(e.gold_ids) for e in exs) / len(exs),
            "steps_bin": dict(Counter(x["steps_bin"] for x in axes)),
            "modality": dict(Counter(x["modality"] for x in axes)),
            "op_family": dict(Counter(x["op_family"] for x in axes)),
            "operators": dict(ops),
            "crosstab_steps_x_modality": {
                s: {m: sum(1 for x in axes if x["steps_bin"] == s and x["modality"] == m)
                    for m in MOD_ORDER} for s in STEP_ORDER},
        }
        t = stats[split]
        print(f"{split}: {t['n_questions']} q | {t['n_facts_per_doc_mean']:.1f} facts/doc | "
              f"~{t['tokens_per_doc_mean']:.0f} tokens/doc | steps {t['steps_bin']} | modality {t['modality']}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(stats, open(a.out, "w"), indent=2)

    # ---- figure: (a) steps by modality (test), (b) operator-family share --------
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 2.9))
    ct = stats["test"]["crosstab_steps_x_modality"]
    w, x0 = 0.26, range(len(STEP_ORDER))
    for i, m in enumerate(MOD_ORDER):
        vals = [ct[s][m] for s in STEP_ORDER]
        bars = ax[0].bar([x + (i - 1) * w for x in x0], vals, w, label=m, color=COL[m])
        for b, v in zip(bars, vals):
            ax[0].text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha="center", fontsize=7)
    ax[0].set_xticks(list(x0)); ax[0].set_xticklabels(STEP_ORDER)
    ax[0].set_xlabel("program steps"); ax[0].set_ylabel("test questions")
    ax[0].set_title("(a) Steps by evidence modality (test)", fontsize=9)
    ax[0].legend(title="gold evidence", fontsize=7, title_fontsize=7, frameon=False)

    fams = ["add_sub_mul", "ratio", "comparison", "table_agg"]
    fam_n = [stats["test"]["op_family"].get(f, 0) for f in fams]
    ax[1].barh(fams[::-1], fam_n[::-1], color="#8172B3")
    for yi, v in enumerate(fam_n[::-1]):
        ax[1].text(v + 8, yi, f"{v} ({100 * v / stats['test']['n_questions']:.0f}%)", va="center", fontsize=7)
    ax[1].set_xlim(0, max(fam_n) * 1.25)
    ax[1].set_xlabel("test questions"); ax[1].set_title("(b) Operator family (test)", fontsize=9)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig), exist_ok=True)
    fig.savefig(a.fig)
    print("wrote", a.out, "and", a.fig)


if __name__ == "__main__":
    main()
