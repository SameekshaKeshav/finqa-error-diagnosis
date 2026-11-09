import pytest

from conftest import make_example
from finqadiag import dsl, grounding, selectors, taxonomy
from finqadiag.data import row_to_text


def test_row_to_text_is_label_free_and_matches_dataset_style():
    ex = make_example()
    assert row_to_text(ex.table, 3) == (
        "company the american express of payments volume ( billions ) is 637 ; "
        "the american express of total transactions ( billions ) is 5.0 ;")
    assert row_to_text(ex.table, 0) == "company ; payments volume ( billions ) ; total transactions ( billions )"


def test_fact_ids_and_text():
    ex = make_example()
    assert ex.fact_ids() == ["text_0", "text_1", "text_2", "table_0", "table_1", "table_2", "table_3"]
    assert ex.fact_text("text_2") == "cards were 86 million ."


def test_candidates_numbers_and_row_labels_with_offsets():
    ex = make_example(program="table_sum(visa inc., none)", gold_ids=("table_1",))
    q, facts, cands = grounding.build_input(ex, ["text_0", "table_1"], sep="</s>")
    assert "</s>" in facts
    for c in cands:  # offsets point at the candidate's own surface string
        assert facts[c.start:c.end] == c.text
    kinds = {(c.kind, c.text) for c in cands}
    assert ("row", "visa inc.") in kinds and ("num", "637") in kinds and ("num", "2457") in kinds


def test_gold_targets_marginalise_over_duplicates():
    ex = make_example(program="divide(637, const_5)", gold_ids=("text_0", "table_3"))
    _, _, cands = grounding.build_input(ex, ["text_0", "table_3"])
    targets, grounded = grounding.gold_token_targets(ex.steps, cands)
    assert grounded
    cand_targets = [t for t in targets if t[0] == "cand"]
    assert len(cand_targets) == 1 and len(cand_targets[0][1]) == 2      # "637" appears twice
    assert targets[-1] == ("res", grounding.RESERVED_IDX[dsl.EOF])


def test_ungrounded_when_operand_missing():
    ex = make_example(program="divide(637, const_5)")
    _, _, cands = grounding.build_input(ex, ["text_1"])                 # no 637 in this fact
    targets, grounded = grounding.gold_token_targets(ex.steps, cands)
    assert not grounded and any(t[0] == "miss" for t in targets)
    assert grounding.operand_coverage(ex, ["text_1"]) == (0, 1)
    assert grounding.operand_coverage(ex, ["table_3"]) == (1, 1)


def test_ordered_fact_ids_is_label_free():
    ex = make_example()
    assert grounding.ordered_fact_ids(ex, ["table_3", "text_2", "text_0"]) == ["text_0", "text_2", "table_3"]


# ---- taxonomy ----------------------------------------------------------------

def tag(pred, ex, retrieved, **kw):
    steps = None if pred is None else dsl.parse_program(pred)
    return taxonomy.tag_example(ex, steps, retrieved, **kw)


def test_gold_axes():
    ex = make_example(program="subtract(5, 1), divide(#0, 5)", gold_ids=("table_3", "text_0"))
    ax = taxonomy.gold_axes(ex)
    assert ax["modality"] == "hybrid" and ax["steps_bin"] == "2" and ax["complexity"] == "multi-step"
    assert ax["op_family"] == "ratio" and ax["n_gold_bin"] == "2"


def test_correct_prediction():
    ex = make_example()
    t = tag("divide(637, const_5)", ex, ["table_3"])
    assert t["correct_ea"] and t["correct_pa"] and t["error_type"] == "correct" and t["locus"] == "correct"


def test_operand_vs_structure_vs_nonexecutable():
    ex = make_example()
    assert tag("divide(2457, const_5)", ex, ["table_3"])["error_type"] == "operand_error"
    assert tag("multiply(637, const_5)", ex, ["table_3"])["error_type"] == "structure_error"
    assert tag(None, ex, ["table_3"])["error_type"] == "non_executable"
    assert tag("divide(637, 0)", ex, ["table_3"])["error_type"] != "correct"


def test_right_answer_wrong_program_counts_as_ea_not_pa():
    ex = make_example(program="divide(637, const_5)", exe_ans=127.4)
    t = tag("multiply(637, 0.2)", ex, ["table_3"])          # same answer, different computation
    assert t["correct_ea"] and not t["correct_pa"]


def test_pa_treats_constant_and_equal_copied_number_as_same_program():
    # Documented leniency: const_5 and a copied "5" denote the same value.
    ex = make_example(program="divide(637, const_5)", exe_ans=127.4)
    assert tag("divide(637, 5)", ex, ["table_3"])["correct_pa"]


def test_locus_attribution_with_and_without_oracle():
    ex = make_example(gold_ids=("table_3",))
    wrong = "divide(2457, const_5)"
    assert tag(wrong, ex, ["table_3"])["locus"] == "generation"            # evidence was there
    assert tag(wrong, ex, ["text_1"])["locus"] == "retrieval_or_both"      # no oracle run
    ok = dsl.parse_program("divide(637, const_5)")
    assert tag(wrong, ex, ["text_1"], oracle_pred_steps=ok, have_oracle=True)["locus"] == "retrieval"
    bad = dsl.parse_program("divide(2457, const_5)")
    assert tag(wrong, ex, ["text_1"], oracle_pred_steps=bad, have_oracle=True)["locus"] == "both"


def test_operands_grounded_can_hold_when_fact_missed():
    # 637 also appears in text_0, so missing table_3 still leaves the operand copyable
    ex = make_example(gold_ids=("table_3",))
    t = tag("divide(2457, const_5)", ex, ["text_0"])
    assert t["retrieval"] == "miss" and t["operands_grounded"] is True


def test_retrieval_breakdown_and_chance():
    ex = make_example(gold_ids=("table_3", "text_0"))
    ranked = {ex.id: ["table_3", "text_1", "text_0", "table_1"]}
    r1 = taxonomy.retrieval_breakdown([ex], ranked, "modality", k=1)["hybrid"]
    r3 = taxonomy.retrieval_breakdown([ex], ranked, "modality", k=3)["hybrid"]
    assert r1["full_recall"] == 0 and r1["fact_recall"] == 0.5
    assert r3["full_recall"] == 1.0
    # 7 facts, 2 gold, k=2 -> exactly 1 of C(7,2)=21 subsets
    assert taxonomy.chance_full_recall([ex], 2) == pytest.approx(1 / 21)


def test_breakdown_aggregates():
    ex1, ex2 = make_example(), make_example()
    ex2.id = "T/1"
    tags = [tag("divide(637, const_5)", ex1, ["table_3"]), tag("divide(2457, const_5)", ex2, ["table_3"])]
    b = taxonomy.breakdown(tags, "modality")["table"]
    assert b["n"] == 2 and b["ea"] == 0.5 and b["n_wrong"] == 1
    assert b["error_type"] == {"operand_error": 1} and b["locus"] == {"generation": 1}


# ---- selectors ---------------------------------------------------------------

def test_selectors():
    ex = make_example(gold_ids=("table_3", "text_0"))
    ranked = {ex.id: [("text_1", 3.0), ("table_1", 2.0), ("table_3", 1.0), ("text_0", 0.5), ("table_2", 0.1)]}
    assert selectors.topk_selector(ranked, 2)(ex) == ["text_1", "table_1"]
    assert set(selectors.train_selector(ranked, 2)(ex)) == {"text_1", "table_1", "table_3", "text_0"}
    o = selectors.oracle_selector(ranked, 3)(ex)
    assert set(o) >= {"table_3", "text_0"} and len(o) == 3 and "text_1" in o
