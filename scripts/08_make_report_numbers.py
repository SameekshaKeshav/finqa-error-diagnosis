"""Generate report/numbers.tex (number macros) and report/table_*.tex from the
results/*.json files, so every number in the report is traceable to a script
output rather than typed by hand.

Run after scripts 01, 01b, 02, 03 and 04.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_split  # noqa: E402
from finqadiag.grounding import operand_coverage  # noqa: E402
from finqadiag.taxonomy import gold_axes, retrieval_status  # noqa: E402

R = "results"


def J(name):
    return json.load(open(os.path.join(R, name)))


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def baseline_tables(m):
    """Trained-baseline tables from runs/ (present only after the GPU run)."""
    from finqadiag.selectors import load_ranked
    from finqadiag.taxonomy import retrieval_breakdown
    from finqadiag.train import evaluate_generator
    if not os.path.exists("runs/gen_released/pred_test.json"):
        return
    D = {s: load_split(f"data/raw/{s}.json") for s in ("dev", "test")}
    res = {}
    for name in ("released", "mine"):
        for split in ("dev", "test"):
            for tag, suf in (("", ""), ("oracle", "_oracle")):
                pr = json.load(open(f"runs/gen_{name}/pred_{split}{suf}.json"))
                res[f"{name}_{split}_{tag or 'top5'}"] = evaluate_generator(pr, D[split])
    mine = load_ranked("runs/retriever/ranked_test.json")
    ids = {k: [f for f, _ in v] for k, v in mine.items()}
    rm = retrieval_breakdown(D["test"], ids, "modality", k=5)
    ra = retrieval_breakdown(D["test"], ids, "all", k=5)["all"]
    json.dump({k: list(v) for k, v in res.items()}, open("results/baseline_summary.json", "w"), indent=1)
    f1 = lambda x: f"{100 * x:.1f}"
    rows = []
    for name, label in (("released", "Released retriever"), ("mine", "My retriever")):
        for tag, tl in (("top5", "top-5 evidence"), ("oracle", "gold evidence")):
            ea, pa = res[f"{name}_test_{tag}"]
            dea, dpa = res[f"{name}_dev_{tag}"]
            rows.append(f"{label} + {tl} & {f1(dea)} & {f1(dpa)} & {f1(ea)} & {f1(pa)} \\\\")
    with open("report/table_baseline.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n & \\multicolumn{2}{c}{Dev} & \\multicolumn{2}{c}{Test} \\\\\n"
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nSetting & EA & PA & EA & PA \\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # error breakdown by modality / steps (released retrieval), from the taxonomy json
    tx = {n: json.load(open(f"results/taxonomy_{n}.json")) for n in ("released", "mine")}
    lines = []
    for ax, order, title in (("modality", ["table", "text", "hybrid"], "Gold evidence"),
                             ("steps_bin", ["1", "2", "3+"], "Program steps")):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for c in order:
            v = tx["released"]["by_axis"][ax][c]
            et = v["error_type"]
            lines.append(f"\\quad {c} & {v['n']} & {f1(v['ea'])} & {f1(v['retrieval_complete'])} & "
                         f"{et.get('structure_error', 0)} & {et.get('operand_error', 0)} \\\\")
    with open("report/table_taxonomy.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nCategory (test) & $n$ & EA & Retr.\\ complete & Structure err. & Operand err. \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    L = tx["released"]["locus"]
    wrong = L.get("generation", 0) + L.get("both", 0) + L.get("retrieval", 0)
    m.update(RelTestEA=f1(res["released_test_top5"][0]), RelTestPA=f1(res["released_test_top5"][1]),
             RelOracleEA=f1(res["released_test_oracle"][0]), RelOraclePA=f1(res["released_test_oracle"][1]),
             MineTestEA=f1(res["mine_test_top5"][0]), MineTestPA=f1(res["mine_test_top5"][1]),
             MineOracleEA=f1(res["mine_test_oracle"][0]),
             MineFull=f1(ra["full_recall"]), MineFactRecall=f1(ra["fact_recall"]),
             MineHybrid=f1(rm["hybrid"]["full_recall"]), MineTable=f1(rm["table"]["full_recall"]),
             MineText=f1(rm["text"]["full_recall"]),
             NWrong=wrong, NGenOnly=L.get("generation", 0), NBoth=L.get("both", 0), NRetOnly=L.get("retrieval", 0),
             PctGenOnly=f1(L.get("generation", 0) / wrong), PctBoth=f1(L.get("both", 0) / wrong),
             PctRetOnly=f1(L.get("retrieval", 0) / wrong))
    ba = tx["released"]["by_axis"]["steps_bin"]
    m["EAThreePlus"] = f1(ba["3+"]["ea"]); m["EAOneStep"] = f1(ba["1"]["ea"])
    m["EAHybrid"] = f1(tx["released"]["by_axis"]["modality"]["hybrid"]["ea"])
    m["EATable"] = f1(tx["released"]["by_axis"]["modality"]["table"]["ea"])
    c = tx["released"]["by_axis"]["retrieval"]["complete"]
    m["NCompleteWrong"] = c["n_wrong"]; m["NStructCompl"] = c["error_type"].get("structure_error", 0)
    m["NOperandCompl"] = c["error_type"].get("operand_error", 0)


def main():
    ds, ret, byk = J("dataset_stats.json"), J("retrieval_released.json"), J("retrieval_released_by_k.json")
    pre, dsl = J("preprocessing_validation.json"), J("dsl_validation.json")
    m = {}

    # ---- dataset ---------------------------------------------------------
    for s, key in (("train", "Train"), ("dev", "Dev"), ("test", "Test")):
        m[f"N{key}"] = ds[s]["n_questions"]
    m["NTotal"] = sum(ds[s]["n_questions"] for s in ("train", "dev", "test"))
    tot = m["NTotal"]
    m["FactsPerDoc"] = f"{sum(ds[s]['n_facts_per_doc_mean'] * ds[s]['n_questions'] for s in ds) / tot:.1f}"
    m["TokensPerDoc"] = f"{sum(ds[s]['tokens_per_doc_mean'] * ds[s]['n_questions'] for s in ds) / tot:.0f}"
    t = ds["test"]
    for mod in ("table", "text", "hybrid"):
        m[f"PctMod{mod.capitalize()}Test"] = pct(t["modality"][mod] / t["n_questions"], 0)
    for b, name in (("1", "One"), ("2", "Two"), ("3+", "ThreePlus")):
        m[f"PctSteps{name}Test"] = pct(t["steps_bin"][b] / t["n_questions"], 0)
    m["PctRatioTest"] = pct(t["op_family"]["ratio"] / t["n_questions"], 0)

    # ---- preprocessing / DSL validation --------------------------------------
    m["DslOk"] = sum(dsl[s]["exec_matches_gold"] for s in dsl)
    m["DslTotal"] = sum(dsl[s]["n"] for s in dsl)
    ser, nser = pre["row_serialisation_vs_dataset"], pre["row_serialisation_total"]
    m["SerTotal"] = nser
    m["SerExactPct"] = pct(ser.get("exact", 0) / nser)
    m["SerWsPct"] = pct(ser.get("whitespace_only", 0) / nser)
    m["SerHeaderPct"] = pct(ser.get("header_row_differs", 0) / nser)
    m["SerOther"] = ser.get("other", 0)
    cov = pre["operand_coverage"]
    m["OperandCovAll"] = pct(sum(cov[s]["all_facts"]["operand_coverage"] * ds[s]["n_questions"] for s in cov) / tot)
    m["ProgCovAll"] = pct(sum(cov[s]["all_facts"]["programs_fully_grounded"] * ds[s]["n_questions"] for s in cov) / tot)
    m["GroundedTest"] = pct(cov["test"]["released_top5"]["programs_fully_grounded"])

    # ---- released-retriever analysis (test) -----------------------------------
    rt = ret["test"]
    m["RelFull"] = pct(rt["overall"]["full_recall"])
    m["RelFact"] = pct(rt["overall"]["fact_recall"])
    m["RelChance"] = pct(rt["chance_full_recall_top5"])
    bm = rt["by_axis"]["modality"]
    for mod in ("table", "text", "hybrid"):
        m[f"Rel{mod.capitalize()}"] = pct(bm[mod]["full_recall"])
        m[f"RelFact{mod.capitalize()}"] = pct(bm[mod]["fact_recall"])
    m["NFailTotal"] = byk["top5_failures_total"]
    for mod in ("table", "text", "hybrid"):
        m[f"NFail{mod.capitalize()}"] = byk["top5_failures_by_modality"][mod]
        m[f"FailShare{mod.capitalize()}"] = pct(byk["failure_share"][mod], 0)
        m[f"QShare{mod.capitalize()}"] = pct(byk["question_share"][mod], 0)
    m["RelAtTwoAll"] = pct(byk["full_recall_by_k"]["all"][1])
    m["RelAtThreeAll"] = pct(byk["full_recall_by_k"]["all"][2])

    # evidence incomplete but operands still copyable (test, released top-5)
    exs = load_split("data/raw/test.json")
    inc = inc_grounded = 0
    for e in exs:
        ids = [f for f, _ in e.released]
        if retrieval_status(e.gold_ids, ids) != "complete":
            inc += 1
            g, n = operand_coverage(e, ids)
            inc_grounded += g == n
    m["NIncomplete"] = inc
    m["NIncompleteGrounded"] = inc_grounded
    m["PctIncompleteGrounded"] = pct(inc_grounded / inc, 0)

    with open("report/numbers.tex", "w") as f:
        f.write("% AUTO-GENERATED by scripts/08_make_report_numbers.py -- do not edit\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # ---- Table 1: dataset ----------------------------------------------------------
    rows = []
    for s, name in (("train", "Train"), ("dev", "Dev"), ("test", "Test")):
        d = ds[s]
        n = d["n_questions"]
        rows.append(" & ".join([
            name, f"{n:,}", f"{d['n_facts_per_doc_mean']:.1f}", f"{d['tokens_per_doc_mean']:.0f}",
            *[pct(d["steps_bin"][b] / n, 0) for b in ("1", "2", "3+")],
            *[pct(d["modality"][x] / n, 0) for x in ("table", "text", "hybrid")]]) + r" \\")
    with open("report/table_dataset.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
                " & & & & \\multicolumn{3}{c}{\\% by program steps} & \\multicolumn{3}{c}{\\% by gold evidence} \\\\\n"
                "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
                "Split & Questions & Facts/doc & Tokens/doc & 1 & 2 & 3+ & table & text & hybrid \\\\\n\\midrule\n"
                + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table 2: retrieval by category (test) ------------------------------------
    def block(title, axis, order, label):
        out = [f"\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for c in order:
            v = rt["by_axis"][axis][c]
            out.append(f"\\quad {label(c)} & {v['n']} & {pct(v['full_recall'])} & {pct(v['fact_recall'])} \\\\")
        return out

    lines = [f"All questions & {rt['n']} & {pct(rt['overall']['full_recall'])} & {pct(rt['overall']['fact_recall'])} \\\\",
             "\\midrule"]
    lines += block("Gold evidence modality", "modality", ["table", "text", "hybrid"], lambda c: c)
    lines += block("Program steps", "steps_bin", ["1", "2", "3+"], lambda c: c)
    lines += block("Operator family", "op_family", ["add_sub_mul", "ratio", "comparison", "table_agg"],
                   lambda c: c.replace("_", "\\_"))
    lines.append(f"\\midrule\nRandom top-5 (chance) & & {pct(rt['chance_full_recall_top5'])} & \\\\")
    with open("report/table_retrieval.tex", "w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n"
                "Category (test) & $n$ & Full recall@5 & Fact recall@5 \\\\\n\\midrule\n"
                + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    baseline_tables(m)
    with open("report/numbers.tex", "w") as f:
        f.write("% AUTO-GENERATED by scripts/08_make_report_numbers.py -- do not edit\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("wrote report/numbers.tex, report/table_dataset.tex, report/table_retrieval.tex")
    print({k: m[k] for k in ("RelFull", "RelTable", "RelText", "RelHybrid", "FailShareHybrid",
                              "QShareHybrid", "GroundedTest", "NIncomplete", "NIncompleteGrounded",
                              "PctIncompleteGrounded", "DslOk", "DslTotal")})


if __name__ == "__main__":
    main()
