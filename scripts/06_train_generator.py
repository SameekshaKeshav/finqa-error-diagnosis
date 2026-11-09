"""Train the program generator, then write predictions used by the taxonomy.

--retrieval released        use the retriever outputs shipped in the FinQA json files
--retrieval runs/retriever  use ranked_{split}.json produced by 05_train_retriever.py

Outputs in --out:
  generator.pt, generator_history.json
  pred_{dev,test}.json          predictions from the retriever's top-k facts
  pred_{dev,test}_oracle.json   predictions when the gold facts are handed over
                                (counterfactual used to attribute failures)

Usage:
  python scripts/06_train_generator.py --model roberta-base --retrieval released --out runs/gen_released
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.generator import ProgramGenerator, build_items, predict  # noqa: E402
from finqadiag.selectors import (load_ranked, oracle_selector, released_ranked,  # noqa: E402
                                 topk_selector, train_selector)
from finqadiag.train import (evaluate_generator, get_device, load_backbone,  # noqa: E402
                             train_generator)


def get_ranked(spec, split, examples):
    if spec == "released":
        return released_ranked(examples)
    return load_ranked(os.path.join(spec, f"ranked_{split}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--retrieval", default="released")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="runs/generator")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_dec", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N examples per split")
    ap.add_argument("--predict_only", action="store_true", help="skip training, load generator.pt")
    a = ap.parse_args()

    D = load_all(a.data)
    if a.limit:
        D = {k: v[:a.limit] for k, v in D.items()}
    ranked = {s: get_ranked(a.retrieval, s, D[s]) for s in ("train", "dev", "test")}
    enc, tok = load_backbone(a.model)
    model = ProgramGenerator(enc, dim=a.dim)

    k = a.topk
    dev_items = build_items(tok, D["dev"], topk_selector(ranked["dev"], k), a.max_len, with_targets=False)
    os.makedirs(a.out, exist_ok=True)
    if not a.predict_only:
        train_items = build_items(tok, D["train"], train_selector(ranked["train"], k), a.max_len,
                                  with_targets=True, drop_ungrounded=True)
        print(f"training items: {len(train_items)}/{len(D['train'])} "
              f"(dropped {len(D['train']) - len(train_items)} whose operands are not copyable)")
        best = train_generator(model, tok, train_items, D["dev"], dev_items, a.out, epochs=a.epochs,
                               lr_enc=a.lr_enc, lr_dec=a.lr_dec, batch_size=a.bs,
                               grad_accum=a.grad_accum, fp16=not a.no_fp16, seed=a.seed)
        print(f"best dev EA: {best:.4f}")

    device = get_device()
    model.load_state_dict(torch.load(os.path.join(a.out, "generator.pt"), map_location=device))
    model.to(device)
    for split in ("dev", "test"):
        for name, sel in (("", topk_selector(ranked[split], k)), ("_oracle", oracle_selector(ranked[split], k))):
            items = build_items(tok, D[split], sel, a.max_len, with_targets=False)
            preds = predict(model, tok, items, device)
            json.dump(preds, open(os.path.join(a.out, f"pred_{split}{name}.json"), "w"))
            ea, pa = evaluate_generator(preds, D[split])
            print(f"{split}{name}: EA {ea:.4f}  PA {pa:.4f}")


if __name__ == "__main__":
    main()
