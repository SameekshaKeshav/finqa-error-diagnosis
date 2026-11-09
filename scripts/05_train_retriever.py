"""Train the fact retriever and write fact rankings for every split.

Outputs in --out:
  retriever.pt, retriever_history.json      best checkpoint (by dev top-5 full recall)
  ranked_{train,dev,test}.json              {example id: [[fact_id, score], ...]} best first

Needs a GPU for the real run (roberta-base ~20-30 min on a Colab T4).

Usage:
  python scripts/05_train_retriever.py --model roberta-base --out runs/retriever
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.retriever import FactScorer, rank_facts  # noqa: E402
from finqadiag.selectors import save_ranked  # noqa: E402
from finqadiag.taxonomy import retrieval_breakdown  # noqa: E402
from finqadiag.train import get_device, load_backbone, train_retriever  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--out", default="runs/retriever")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg_ratio", type=float, default=3.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N examples per split")
    ap.add_argument("--rank_splits", nargs="+", default=["dev", "test", "train"])
    a = ap.parse_args()

    D = load_all(a.data)
    if a.limit:
        D = {k: v[:a.limit] for k, v in D.items()}
    enc, tok = load_backbone(a.model)
    model = FactScorer(enc)
    best = train_retriever(model, tok, D["train"], D["dev"], a.out, epochs=a.epochs, lr=a.lr,
                           batch_size=a.bs, neg_ratio=a.neg_ratio, max_len=a.max_len,
                           fp16=not a.no_fp16, seed=a.seed)
    print(f"best dev top-5 full recall: {best:.4f}")

    device = get_device()
    model.load_state_dict(torch.load(os.path.join(a.out, "retriever.pt"), map_location=device))
    model.to(device)
    for split in a.rank_splits:
        ranked = rank_facts(model, tok, D[split], device)
        save_ranked(os.path.join(a.out, f"ranked_{split}.json"), ranked)
        r = retrieval_breakdown(D[split], {k: [f for f, _ in v] for k, v in ranked.items()}, "modality", k=5)
        print(split, {m: round(v["full_recall"], 3) for m, v in r.items()})


if __name__ == "__main__":
    main()
