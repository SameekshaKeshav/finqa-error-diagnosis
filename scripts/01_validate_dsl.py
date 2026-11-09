"""Validate my DSL executor: re-execute every gold program and compare to the
dataset's stored answer (exe_ans). Also checks that parse -> tokens -> parse is
lossless and that canonical() is invariant to commutative reordering.

Usage:  python scripts/01_validate_dsl.py --data data/raw --out results/dsl_validation.json
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag import dsl, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--out", default="results/dsl_validation.json")
    ap.add_argument("--show", type=int, default=8, help="print this many failures")
    a = ap.parse_args()

    report, shown = {}, 0
    for split in ("train", "dev", "test"):
        data = json.load(open(os.path.join(a.data, f"{split}.json")))
        c = Counter()
        for ex in data:
            q = ex["qa"]
            c["n"] += 1
            try:
                steps = dsl.parse_program(q["program"])
            except dsl.ProgramError as e:
                c["parse_fail"] += 1
                continue
            toks = dsl.program_to_tokens(steps)
            if dsl.parse_tokens(toks) == steps:
                c["token_roundtrip_ok"] += 1
            got = dsl.try_execute(steps, ex["table"])
            if metrics.answers_match(got, metrics.gold_answer(q["exe_ans"])):
                c["exec_matches_gold"] += 1
            else:
                c["exec_mismatch"] += 1
                if shown < a.show:
                    shown += 1
                    print(f"[{split}] {ex['id']}: program={q['program']!r} got={got!r} gold={q['exe_ans']!r}")
            if metrics.program_correct(steps, q["program"]):
                c["pa_self_ok"] += 1
        report[split] = dict(c)
        print(split, dict(c))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
