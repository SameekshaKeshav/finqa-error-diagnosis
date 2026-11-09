"""Integration test on the real dev split using SYNTHETIC predictions
(gold programs with deliberately injected errors).  It verifies the plumbing and
the accounting of the taxonomy; its numbers are test fixtures, never results."""
import json
import os
import subprocess
import sys

import pytest

from finqadiag import dsl
from finqadiag.data import load_split
from finqadiag.selectors import oracle_selector, released_ranked, topk_selector
from finqadiag.taxonomy import breakdown, tags_from_predictions

DEV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dev.json")
pytestmark = pytest.mark.skipif(not os.path.exists(DEV), reason="FinQA data not downloaded")


def _corrupt(steps):
    """Same program but the first non-reference operand replaced -> operand_error."""
    out = []
    done = False
    for s in steps:
        args = list(s.args)
        if not done:
            for i, a in enumerate(args):
                if not a.startswith("#") and not a.startswith("const_") and a != "none":
                    args[i] = "987654.321"
                    done = True
                    break
        out.append(dsl.Step(s.op, tuple(args)))
    return out


def test_taxonomy_accounting_on_real_dev_split():
    exs = load_split(DEV)
    ranked = released_ranked(exs)
    sel, osel = topk_selector(ranked, 5), oracle_selector(ranked, 5)
    preds, oracle = {}, {}
    for i, ex in enumerate(exs):
        good = dsl.program_to_tokens(ex.steps)
        bad = dsl.program_to_tokens(_corrupt(ex.steps))
        preds[ex.id] = {"tokens": good if i % 3 else bad, "facts": list(sel(ex))}
        oracle[ex.id] = {"tokens": good, "facts": list(osel(ex))}
    tags = tags_from_predictions(exs, preds, oracle)

    assert len(tags) == len(exs)
    wrong = [t for t in tags if not t["correct_ea"]]
    assert 0.25 < len(wrong) / len(tags) < 0.40                # ~1/3 corrupted (some corruptions may still pass)
    assert all(t["error_type"] in ("operand_error", "non_executable", "structure_error") for t in wrong)
    # with a perfect oracle, an incomplete retrieval means the failure is attributed to retrieval
    for t in wrong:
        if t["retrieval"] != "complete":
            assert t["locus"] == "retrieval"
        else:
            assert t["locus"] == "generation"
    # every example lands in exactly one modality bucket and counts add up
    by_mod = breakdown(tags, "modality")
    assert sum(v["n"] for v in by_mod.values()) == len(exs)
    assert sum(v["n_wrong"] for v in by_mod.values()) == len(wrong)


def test_report_script_runs_end_to_end(tmp_path):
    exs = load_split(DEV)[:120]
    ranked = released_ranked(exs)
    sel = topk_selector(ranked, 5)
    preds = {ex.id: {"tokens": dsl.program_to_tokens(ex.steps), "facts": list(sel(ex))} for ex in exs}
    pred_path = tmp_path / "pred.json"
    json.dump(preds, open(pred_path, "w"))
    # script loads the full split, so give it a data dir holding a trimmed dev.json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    full = json.load(open(DEV))
    keep = {ex.id for ex in exs}
    for split in ("train", "dev", "test"):
        json.dump([r for r in full if r["id"] in keep] if split == "dev" else full[:1],
                  open(data_dir / f"{split}.json", "w"))
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "07_taxonomy_report.py")
    r = subprocess.run([sys.executable, script, "--data", str(data_dir), "--split", "dev",
                        "--pred", str(pred_path), "--name", "t"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "results" / "taxonomy_t.json").exists()
    assert (tmp_path / "report" / "figures" / "taxonomy_t.pdf").exists()
    summary = json.load(open(tmp_path / "results" / "taxonomy_t.json"))
    assert summary["ea"] == 1.0 and summary["pa"] == 1.0
