"""[GPU] Beam-decode dev and test with a trained generator checkpoint.

Usage (checkpoint produced by 06_train_generator.py):
  python scripts/09_beam_decode.py --ckpt runs/gen_released --retrieval released --beam 8
Writes <ckpt>/beams_{dev,test}.json  ({id: {"beams": [{"tokens", "logp"}], "facts": [...]}}).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finqadiag.data import load_all  # noqa: E402
from finqadiag.generator import ProgramGenerator, build_items, predict_beam  # noqa: E402
from finqadiag.selectors import load_ranked, released_ranked, topk_selector  # noqa: E402
from finqadiag.train import get_device, load_backbone  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--ckpt", required=True, help="dir containing generator.pt")
    ap.add_argument("--retrieval", default="released")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    D = load_all(a.data)
    if a.limit:
        D = {k: v[:a.limit] for k, v in D.items()}
    enc, tok = load_backbone(a.model)
    model = ProgramGenerator(enc, dim=a.dim)
    device = get_device()
    model.load_state_dict(torch.load(os.path.join(a.ckpt, "generator.pt"), map_location=device))
    model.to(device)
    for split in ("dev", "test"):
        ranked = released_ranked(D[split]) if a.retrieval == "released" else \
            load_ranked(os.path.join(a.retrieval, f"ranked_{split}.json"))
        items = build_items(tok, D[split], topk_selector(ranked, a.topk), a.max_len, with_targets=False)
        preds = predict_beam(model, tok, items, device, beam=a.beam)
        json.dump(preds, open(os.path.join(a.ckpt, f"beams_{split}.json"), "w"))
        print(split, "decoded", len(preds), "questions, beam", a.beam, flush=True)


if __name__ == "__main__":
    main()
