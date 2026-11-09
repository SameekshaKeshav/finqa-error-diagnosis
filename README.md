# Diagnosing and Repairing Multi-Step Reasoning Failures in FinQA

Midterm deliverable, ECE592/CSC591 — Sameeksha Keshav (skeshav2@ncsu.edu). Solo project.

## What is mine and what is not

**All code in `src/`, `scripts/`, `tests/` is written by me.** I did not copy or adapt any open-source
FinQA implementation. The FinQANet *design* (retriever + program generator) follows the FinQA paper
(Chen et al., EMNLP 2021); the implementation is my own.

Third-party material used:

| Item | Source | Use |
|---|---|---|
| FinQA `train/dev/test.json` | https://github.com/czyssrs/FinQA (public dataset) | data only (`scripts/00_download_data.sh`) |
| Released retriever rankings (`retriever_*` fields inside the same JSON files) | FinQA release | analysed in `scripts/02`, `04`; used as a fixed selector. I do not know which checkpoint produced them. |
| RoBERTa weights, `transformers`, `torch` | Hugging Face / PyTorch libraries | pretrained encoder; library code only |

## Status (final)

* Phase 1 and 2 done; Phase 3 (MultiHiertt) not carried out.
* Four released-retrieval generator runs (`roberta-base`): test EA 60.6 +- 0.7, PA 58.7 +- 0.6 (`results/phase2_analysis.json`); gold evidence changes EA by under 0.5 points, so retrieval is not the bottleneck.
* Errors are in the program: 71% of 3+ step failures are programs with too few steps (`report/table_mechanisms.tex`).
* Beam search (16) + execution-guided reranking: right answer in the beam for 80.3% of test questions, but reranking gains only +0.4 (voting) / +0.6 (learned) EA over beam top-1, not statistically significant. Voting across 3 independently trained models: 62.4.
* Report: `report/final.tex` (final), `report/milestone.tex` (midterm).
* Reproduce the Phase 2 numbers: GPU run `notebooks/run_phase2_kaggle.ipynb`, then `python scripts/11_learned_rerank.py --beams runs/gen_beam` and `python scripts/12_phase2_analysis.py` (CPU).

## Layout

```
src/finqadiag/
  dsl.py        program parser/executor for the 10-op FinQA DSL, canonical + skeleton forms
  metrics.py    execution accuracy (EA) and program accuracy (PA)
  data.py       Example dataclass, label-free row serialization, split loader
  grounding.py  reserved vocab, number/row-label copy candidates, token targets, operand coverage
  taxonomy.py   gold-derived and prediction-derived error axes, locus via oracle counterfactual
  selectors.py  top-k / train / oracle evidence selectors
  retriever.py  RoBERTa cross-encoder fact scorer
  generator.py  encoder + LSTM pointer decoder with unified softmax and marginal loss; greedy and beam decoding
  rerank.py     execution-guided reranking of beam candidates (answer-level voting), no torch
  learned_rerank.py  features + answer-level selection for the learned reranker
  mechanisms.py classify why a wrong program is wrong (too few steps, wrong operand, ...)
  train.py      training/eval loops for both models
  tiny.py       tiny offline tokenizer/encoder for CPU smoke tests only
scripts/
  00_download_data.sh            fetch the three JSON files
  01_validate_dsl.py             gold programs re-execute to gold answers
  01b_validate_preprocessing.py  serialization vs dataset text; operand coverage
  02_retrieval_analysis.py       released-retriever recall by taxonomy axis
  03_dataset_stats.py            split statistics + figure
  04_plot_retrieval.py           recall@k figure
  05_train_retriever.py          [GPU] train my retriever
  06_train_generator.py          [GPU] train generator, write predictions (+ oracle-evidence run)
  07_taxonomy_report.py          taxonomy tables/figure from predictions
  08_make_report_numbers.py      LaTeX macros/tables from results/*.json
  09_beam_decode.py              [GPU] beam-decode dev/test with a trained checkpoint
  10_rerank_eval.py              beam top-1 vs voting reranker; k/temp chosen on dev; oracle@k; per-axis
  11_learned_rerank.py           learned reranker trained on dev beams, evaluated once on test (scikit-learn)
  12_phase2_analysis.py          error mechanisms, multi-model voting, oracle@k figure, LaTeX numbers/tables
tests/                           61 tests (pytest)
report/milestone.tex             NeurIPS-format midterm report (milestone.pdf)
```

## Run

```bash
pip install -r requirements.txt
bash scripts/00_download_data.sh
python -m pytest -q
python scripts/01_validate_dsl.py
python scripts/01b_validate_preprocessing.py
python scripts/02_retrieval_analysis.py
python scripts/03_dataset_stats.py
python scripts/04_plot_retrieval.py
python scripts/08_make_report_numbers.py
cd report && pdflatex milestone && bibtex milestone && pdflatex milestone && pdflatex milestone
```

GPU steps (see the notebook for exact arguments): `05_train_retriever.py`, `06_train_generator.py`
(`--retrieval released` first, then with my retriever), then `07_taxonomy_report.py`.
