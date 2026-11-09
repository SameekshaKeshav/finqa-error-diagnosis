"""Phase-2 analysis (CPU, no torch): error mechanisms, multi-model voting, oracle@k figure,
LaTeX numbers/tables for the final report.

Needs: runs/{gen_released,gen_released_s1,gen_released_s2,gen_beam}/pred_test.json,
       results/rerank_released.json (script 10), results/learned_rerank_released.json (script 11).
"""
import json
import os
import statistics as st
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_split  # noqa: E402
from finqadiag.dsl import ProgramError, parse_tokens, try_execute  # noqa: E402
from finqadiag.mechanisms import CATEGORIES, mechanism  # noqa: E402
from finqadiag.metrics import answers_match, execution_correct, program_correct  # noqa: E402
from finqadiag.taxonomy import gold_axes  # noqa: E402

RUNS = ["gen_released", "gen_released_s1", "gen_released_s2", "gen_beam"]
f1 = lambda x: f"{100 * x:.1f}"  # noqa: E731


def parse(p, e):
    try:
        return parse_tokens(p[e.id]["tokens"])
    except ProgramError:
        return None


def key(v):
    return None if v is None else (v if isinstance(v, str) else round(float(v), 5))


def main():
    ex = load_split("data/raw/test.json")
    P = {r: json.load(open(f"runs/{r}/pred_test.json")) for r in RUNS}
    S = {r: [parse(P[r], e) for e in ex] for r in RUNS}
    out, m = {}, {}

    # ---- single-model numbers --------------------------------------------------
    ea = {r: sum(bool(execution_correct(s, e.table, e.exe_ans)) for s, e in zip(S[r], ex)) / len(ex) for r in RUNS}
    pa = {r: sum(bool(program_correct(s, e.program)) for s, e in zip(S[r], ex)) / len(ex) for r in RUNS}
    out["single"] = {r: [ea[r], pa[r]] for r in RUNS}
    m["SingleEAMean"] = f1(st.mean(ea.values())); m["SingleEASd"] = f1(st.stdev(ea.values()))
    m["SinglePAMean"] = f1(st.mean(pa.values())); m["SinglePASd"] = f1(st.stdev(pa.values()))
    m["SingleEAMin"] = f1(min(ea.values())); m["SingleEAMax"] = f1(max(ea.values()))
    m["NRuns"] = len(RUNS)

    # ---- voting over independently trained models ------------------------------------
    def vote(members):
        ok = 0
        for i, e in enumerate(ex):
            keys = [key(try_execute(list(S[r][i]), e.table)) if S[r][i] else None for r in members]
            cnt = Counter(k for k in keys if k is not None)
            if not cnt:
                continue
            top = max(cnt.values())
            pick = next(k for k in keys if k is not None and cnt[k] == top)   # ties -> first member
            ok += answers_match(pick, e.exe_ans)
        return ok / len(ex)

    def any_correct(members):
        n = 0
        for i, e in enumerate(ex):
            n += any(S[r][i] is not None and (lambda v: v is not None and answers_match(v, e.exe_ans))(
                try_execute(list(S[r][i]), e.table)) for r in members)
        return n / len(ex)
    out["vote"] = {"3 models": vote(RUNS[:3]), "4 models": vote(RUNS)}
    out["any_correct"] = {"3 models": any_correct(RUNS[:3]), "4 models": any_correct(RUNS)}
    m["VoteThree"] = f1(out["vote"]["3 models"]); m["VoteFour"] = f1(out["vote"]["4 models"])
    m["AnyThree"] = f1(out["any_correct"]["3 models"]); m["AnyFour"] = f1(out["any_correct"]["4 models"])

    # ---- error mechanisms (pooled over the greedy runs) ----------------------------
    mech, mech_by_steps, n_wrong = Counter(), {b: Counter() for b in ("1", "2", "3+")}, 0
    wrong_by_steps = Counter()
    for r in RUNS:
        for s, e in zip(S[r], ex):
            if execution_correct(s, e.table, e.exe_ans):
                continue
            c = mechanism(s, e.steps)
            mech[c] += 1; n_wrong += 1
            b = gold_axes(e)["steps_bin"]
            mech_by_steps[b][c] += 1; wrong_by_steps[b] += 1
    out["mechanisms"] = {"n_wrong_pooled": n_wrong, "counts": dict(mech),
                         "by_steps": {b: dict(c) for b, c in mech_by_steps.items()}}
    for c in CATEGORIES:
        m["Mech" + "".join(w.capitalize() for w in c.split("_"))] = f1(mech[c] / n_wrong)
    m["MechFewManyShare"] = f1((mech["too_few_steps"] + mech["too_many_steps"]) / n_wrong)
    m["NWrongPooled"] = n_wrong
    rows = []
    label = {"unparsable": "not a valid program", "too_few_steps": "too few steps",
             "too_many_steps": "too many steps", "wrong_operator": "wrong operator (same step count)",
             "args_swapped": "arguments swapped", "wrong_constant": "wrong constant",
             "wrong_reference": "wrong step reference", "wrong_operand": "wrong document operand"}
    for c in CATEGORIES:
        cells = [f1(mech[c] / n_wrong)] + [f1(mech_by_steps[b][c] / max(wrong_by_steps[b], 1)) for b in ("1", "2", "3+")]
        rows.append(f"{label[c]} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule " + "\nWrong answers pooled ($n$) & " + " & ".join(
        [str(n_wrong)] + [str(wrong_by_steps[b]) for b in ("1", "2", "3+")]) + r" \\")
    open("report/table_mechanisms.tex", "w").write(
        "\\begin{tabular}{lrrrr}\n\\toprule\nMechanism (\\% of wrong answers) & All & 1 step & 2 steps & 3+ steps \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Phase 2: beam rerank / learned rerank ------------------------------------------------
    rr = json.load(open("results/rerank_released.json"))
    lr = json.load(open("results/learned_rerank_released.json"))
    sel = rr["selected"]
    t = rr["test"]
    k16 = [k for k in t if k.startswith("beam top-1")][0]
    base = t[k16]; mar = t["marginal rerank (dev-selected)"]; lea = lr["test_learned"]
    greedy_beamrun = out["single"]["gen_beam"]
    m.update(BeamTopEA=f1(base[0]), BeamTopPA=f1(base[1]), GreedyBeamRunEA=f1(greedy_beamrun[0]),
             MargEA=f1(mar[0]), MargPA=f1(mar[1]), LearnEA=f1(lea[0]), LearnPA=f1(lea[1]),
             MargK=sel[2], MargT=sel[1], LearnModel=lr["selected"].replace("_", " "),
             LearnFixed=lr["paired"]["fixed"], LearnBroke=lr["paired"]["broke"],
             LearnCIlo=f"{100 * lr['gain_ci95'][0]:.1f}", LearnCIhi=f"{100 * lr['gain_ci95'][1]:.1f}")
    # paired comparison of the voting reranker with beam top-1 (needs the beams files)
    from math import comb
    from finqadiag.rerank import per_example_correct
    Pb = json.load(open("runs/gen_beam/beams_test.json"))
    b1 = per_example_correct(Pb, ex, "top1", 1.0, 1)
    b2 = per_example_correct(Pb, ex, "marginal", sel[1], sel[2])
    fixed = sum(b2[e.id] and not b1[e.id] for e in ex); broke = sum(b1[e.id] and not b2[e.id] for e in ex)
    n = fixed + broke
    pval = min(1.0, 2 * sum(comb(n, i) for i in range(max(fixed, broke), n + 1)) / 2 ** n) if n else 1.0
    m.update(MargFixed=fixed, MargBroke=broke, MargP=f"{pval:.2f}")
    orc = {int(k): v for k, v in rr["oracle_at_k_test"].items()}
    for k, v in orc.items():
        m["OracleK" + {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen"}[k]] = f1(v)
    sa = rr["by_axis_test"]["steps_bin"]
    m["ThreePlusBase"] = f1(sa["3+"]["base_ea"]); m["ThreePlusRerank"] = f1(sa["3+"]["rerank_ea"])
    la = lr["by_axis_test"]["steps_bin"]
    m["ThreePlusLearn"] = f1(la["3+"]["learned_ea"]); m["NThreePlus"] = la["3+"]["n"]
    rows = [
        ("Greedy decoding (same model)", greedy_beamrun[0], greedy_beamrun[1]),
        ("Beam search (16), top-1", base[0], base[1]),
        (f"+ answer voting over beams ($k{{=}}{sel[2]}$, $T{{=}}{sel[1]}$; dev-selected)", mar[0], mar[1]),
        (f"+ learned reranker ({lr['selected']}; dev-trained)", lea[0], lea[1]),
    ]
    tab = "\n".join(f"{n} & {f1(a)} & {f1(b)} \\\\" for n, a, b in rows)
    tab += ("\n\\midrule\n" + f"Voting over {len(RUNS)-1} independently trained models & {f1(out['vote']['3 models'])} & -- \\\\\n"
            + f"Oracle: right answer among top-16 beams & {f1(orc[16])} & -- \\\\\n"
            + f"Oracle: right answer from any of 3 models & {f1(out['any_correct']['3 models'])} & -- \\\\")
    open("report/table_phase2.tex", "w").write(
        "\\begin{tabular}{lrr}\n\\toprule\nTest set, released retrieval & EA & PA \\\\\n\\midrule\n" + tab + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- figure ------------------------------------------------------------------------------
    ks = sorted(orc)
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.plot(ks, [100 * orc[k] for k in ks], "o-", color="#0072B2", lw=1.8)
    ax.text(ks[-1], 100 * orc[ks[-1]] + 1.4, "oracle@$k$: right answer in top-$k$ beams", ha="right", fontsize=8, color="#0072B2")
    ax.axhspan(100 * base[0], 100 * out["vote"]["3 models"], color="#999999", alpha=0.25, lw=0)
    ax.text(ks[-1], 100 * out["vote"]["3 models"] + 1.2,
            "beam top-1, voting, learned rerank,\n3-model vote: %.1f--%.1f%%" % (100 * base[0], 100 * out["vote"]["3 models"]),
            ha="right", va="bottom", fontsize=8, color="#333333")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("beam size $k$"); ax.set_ylabel("test execution accuracy (%)")
    ax.set_ylim(55, 85)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.tight_layout(); os.makedirs("report/figures", exist_ok=True)
    fig.savefig("report/figures/oracle_at_k.pdf")

    json.dump(out, open("results/phase2_analysis.json", "w"), indent=1)
    with open("report/numbers_phase2.tex", "w") as f:
        f.write("% AUTO-GENERATED by scripts/12_phase2_analysis.py\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(json.dumps({k: m[k] for k in m}, indent=0)[:1800])
    print(json.dumps(out["mechanisms"], indent=0)[:900])


if __name__ == "__main__":
    main()
