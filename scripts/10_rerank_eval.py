"""Evaluate beam search and execution-guided reranking (CPU, no torch).

Configuration (k, mode, temp) is chosen on DEV; TEST is evaluated once with it.
Usage: python scripts/10_rerank_eval.py --beams runs/gen_released --name released
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.rerank import evaluate, oracle_at_k, per_example_correct  # noqa: E402
from finqadiag.taxonomy import gold_axes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--beams", required=True)
    ap.add_argument("--name", default="released")
    a = ap.parse_args()
    D = load_all(a.data)
    P = {s: json.load(open(os.path.join(a.beams, f"beams_{s}.json"))) for s in ("dev", "test")}
    D = {s: [e for e in D[s] if e.id in P[s]] for s in ("dev", "test")}   # supports --limit smoke runs
    kmax = max(len(v["beams"]) for v in P["dev"].values())
    ks = [k for k in (1, 2, 4, 8, 16) if k <= kmax]
    grid = [("top1", 1.0, 1)] + [(m, t, k) for m in ("exec_top1",) for t in (1.0,) for k in ks if k > 1] \
        + [("marginal", t, k) for t in (0.5, 1.0, 2.0) for k in ks if k > 1]
    dev = {}
    for mode, t, k in grid:
        dev[(mode, t, k)] = evaluate(P["dev"], D["dev"], mode, t, k)
    best = max((c for c in dev if c[0] == "marginal"), key=lambda c: dev[c][0])
    print("DEV grid (EA, PA):")
    for c, v in sorted(dev.items(), key=lambda x: -x[1][0])[:10]:
        print(f"  {c}: EA {v[0]:.4f} PA {v[1]:.4f}")
    print("selected on dev:", best)
    res = {"selected": list(best), "dev": {str(c): v for c, v in dev.items()}, "test": {}}
    for name, cfg in (("beam top-1 (k=%d)" % kmax, ("top1", 1.0, kmax)), ("greedy-equivalent top-1 (k=1)", ("top1", 1.0, 1)),
                      ("executable top-1", ("exec_top1", 1.0, kmax)), ("marginal rerank (dev-selected)", best)):
        res["test"][name] = evaluate(P["test"], D["test"], cfg[0], cfg[1], cfg[2])
        print(f"TEST {name}: EA {res['test'][name][0]:.4f} PA {res['test'][name][1]:.4f}")
    res["oracle_at_k_test"] = {k: oracle_at_k(P["test"], D["test"], k) for k in ks}
    res["oracle_at_k_dev"] = {k: oracle_at_k(P["dev"], D["dev"], k) for k in ks}
    print("oracle@k test:", {k: round(v, 4) for k, v in res["oracle_at_k_test"].items()})
    # per-axis, test: greedy-equivalent vs rerank
    base = per_example_correct(P["test"], D["test"], "top1", 1.0, 1)
    rer = per_example_correct(P["test"], D["test"], best[0], best[1], best[2])
    axes = {}
    for ex in D["test"]:
        g = gold_axes(ex)
        for ax in ("steps_bin", "modality"):
            d = axes.setdefault(ax, {}).setdefault(g[ax], [0, 0, 0])
            d[0] += 1; d[1] += base[ex.id]; d[2] += rer[ex.id]
    res["by_axis_test"] = {ax: {c: {"n": v[0], "base_ea": v[1] / v[0], "rerank_ea": v[2] / v[0]}
                                for c, v in cs.items()} for ax, cs in axes.items()}
    for ax, cs in res["by_axis_test"].items():
        for c, v in cs.items():
            print(f"  {ax:9s} {c:7s} n={v['n']:4d} base {v['base_ea']:.3f} -> rerank {v['rerank_ea']:.3f}")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/rerank_{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
