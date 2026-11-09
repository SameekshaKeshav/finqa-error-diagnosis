import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finqadiag.data import Example  # noqa: E402
from finqadiag.dsl import parse_program  # noqa: E402
from finqadiag.metrics import gold_answer  # noqa: E402


def make_example(program="divide(637, const_5)", gold_ids=("table_3",), exe_ans=127.4,
                 question="what is the average payment volume per transaction?",
                 pre=("the company reported 637 million in volume .", "other text ."),
                 post=("cards were 86 million .",), table=None, released=None):
    table = table or [
        ["company", "payments volume ( billions )", "total transactions ( billions )"],
        ["visa inc.", "$ 2457", "50.3"],
        ["mastercard", "$ 1500", "30.1"],
        ["american express", "637", "5.0"],
    ]
    return Example(id="T/0", question=question, pre_text=list(pre), post_text=list(post),
                   table=table, program=program, steps=parse_program(program),
                   exe_ans=gold_answer(exe_ans), gold_ids=list(gold_ids), released=released)
