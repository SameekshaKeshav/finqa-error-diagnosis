"""Learned reranker trained on DEV beams, evaluated once on TEST (CPU; needs scikit-learn).

Model/hyper-parameters are chosen by 5-fold cross-validation on dev (grouped by question);
the final model is fit on all of dev.  Usage:
  python scripts/11_learned_rerank.py --beams runs/gen_beam --name released
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.learned_rerank import FEATURES, build_dataset, pick  # noqa: E402
from finqadiag.metrics import execution_correct, program_correct  # noqa: E402
from finqadiag.rerank import _parse  # noqa: E402
from finqadiag.taxonomy import gold_axes  # noqa: E402


def top1_steps(preds, examples):
    return [_parse(preds[e.id]["beams"][0]["tokens"]) if preds[e.id]["beams"] else None for e in examples]


def score(steps, examples):
    ea = np.mean([bool(execution_correct(s, e.table, e.exe_ans)) for s, e in zip(steps, examples)])
    pa = np.mean([bool(program_correct(s, e.program)) for s, e in zip(steps, examples)])
    return float(ea), float(pa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--beams", required=True)
    ap.add_argument("--name", default="released")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = load_all(a.data)
    P = {s: json.load(open(os.path.join(a.beams, f"beams_{s}.json"))) for s in ("dev", "test")}
    dev, test = D["dev"], D["test"]
    Xd, yd, gd, Sd, Kd, Ld = build_dataset(P["dev"], dev)
    Xt, yt, gt, St, Kt, Lt = build_dataset(P["test"], test)
    fb_dev, fb_test = top1_steps(P["dev"], dev), top1_steps(P["test"], test)
    print(f"dev rows {len(yd)} (pos {yd.mean():.3f}) | test rows {len(yt)}")
    models = {
        "logreg C=0.1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "logreg C=1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "gbm d2": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                     subsample=0.8, random_state=a.seed),
        "gbm d3": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                     subsample=0.8, random_state=a.seed),
    }
    cv = {}
    for name, mk in models.items():
        chosen = list(fb_dev)
        for tr, va in GroupKFold(5).split(Xd, yd, gd):
            m = mk().fit(Xd[tr], yd[tr])
            p = m.predict_proba(Xd[va])[:, 1]
            sub = pick(p, gd[va], [Sd[i] for i in va], [Kd[i] for i in va], Ld[va], chosen)
            chosen = sub
        cv[name] = score(chosen, dev)
        print(f"CV on dev  {name:14s} EA {cv[name][0]:.4f} PA {cv[name][1]:.4f}")
    base_dev = score(fb_dev, dev)
    print(f"dev top-1 beam baseline      EA {base_dev[0]:.4f}")
    best = max(cv, key=lambda k: cv[k][0])
    print("selected by dev-CV:", best)
    m = models[best]().fit(Xd, yd)
    chosen = pick(m.predict_proba(Xt)[:, 1], gt, St, Kt, Lt, fb_test)
    res = {"selected": best, "cv_dev": cv, "dev_top1": base_dev,
           "test_top1": score(fb_test, test), "test_learned": score(chosen, test)}
    print(f"TEST top-1 beam   EA {res['test_top1'][0]:.4f} PA {res['test_top1'][1]:.4f}")
    print(f"TEST learned      EA {res['test_learned'][0]:.4f} PA {res['test_learned'][1]:.4f}")
    b = np.array([bool(execution_correct(s, e.table, e.exe_ans)) for s, e in zip(fb_test, test)])
    r = np.array([bool(execution_correct(s, e.table, e.exe_ans)) for s, e in zip(chosen, test)])
    res["paired"] = {"fixed": int((r & ~b).sum()), "broke": int((b & ~r).sum())}
    rng = np.random.default_rng(0)
    d = r.astype(int) - b.astype(int)
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    res["gain_ci95"] = [float(bs[125]), float(bs[4875])]
    print("paired:", res["paired"], "gain 95%% CI (pts): [%.2f, %.2f]" % (100 * bs[125], 100 * bs[4875]))
    axes = {}
    for e, bi, ri in zip(test, b, r):
        g = gold_axes(e)
        for ax in ("steps_bin", "modality"):
            c = axes.setdefault(ax, {}).setdefault(g[ax], [0, 0, 0]); c[0] += 1; c[1] += bi; c[2] += ri
    res["by_axis_test"] = {ax: {c: {"n": v[0], "top1_ea": v[1] / v[0], "learned_ea": v[2] / v[0]}
                                for c, v in cs.items()} for ax, cs in axes.items()}
    for ax, cs in res["by_axis_test"].items():
        for c, v in cs.items():
            print(f"  {ax:9s} {c:7s} n={v['n']:4d} top1 {v['top1_ea']:.3f} -> learned {v['learned_ea']:.3f}")
    if hasattr(m, "coef_") or hasattr(m, "feature_importances_"):
        imp = m[-1].coef_[0] if hasattr(m, "steps") else m.feature_importances_
        top = sorted(zip(FEATURES, imp), key=lambda x: -abs(x[1]))[:8]
        res["top_features"] = [(f, float(w)) for f, w in top]
        print("top features:", [(f, round(float(w), 3)) for f, w in top])
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/learned_rerank_{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
