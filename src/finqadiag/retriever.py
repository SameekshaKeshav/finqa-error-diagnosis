"""Fact retriever: a cross-encoder that scores (question, fact) pairs.

Training pairs: every gold fact is a positive; negatives are sampled uniformly
from the other facts of the same document (``neg_ratio`` per positive,
re-sampled every epoch).  Positives and negatives are serialised by the same
function (data.row_to_text / raw sentence), so nothing about the text reveals
the label.  At inference every fact of a document is scored and the facts are
ranked; the top-k go to the generator.

Own implementation of the standard retriever-generator recipe.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .data import Example


class FactScorer(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        H = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(H, H), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(H, 1))

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)


def make_pairs(examples: Sequence[Example], neg_ratio: float, rng: random.Random
               ) -> List[Tuple[int, str, int]]:
    """(example index, fact id, label) pairs for one epoch."""
    pairs = []
    for i, ex in enumerate(examples):
        gold = set(ex.gold_ids)
        negs = [f for f in ex.fact_ids() if f not in gold]
        k = min(len(negs), max(1, int(round(neg_ratio * len(gold)))))
        for g in gold:
            pairs.append((i, g, 1))
        for f in rng.sample(negs, k):
            pairs.append((i, f, 0))
    rng.shuffle(pairs)
    return pairs


def encode_pairs(tokenizer, questions: Sequence[str], facts: Sequence[str], max_len: int = 256):
    return tokenizer(list(questions), list(facts), truncation="only_second",
                     max_length=max_len, padding=True, return_tensors="pt")


@torch.no_grad()
def rank_facts(model: FactScorer, tokenizer, examples: Sequence[Example], device,
               batch_size: int = 64, max_len: int = 256) -> Dict[str, List[Tuple[str, float]]]:
    """Score every fact of every example; returns {id: [(fact_id, score), ...]} best first."""
    model.eval()
    jobs = [(ex.id, ex.question, fid, ex.fact_text(fid)) for ex in examples for fid in ex.fact_ids()]
    scores: Dict[str, List[Tuple[str, float]]] = {ex.id: [] for ex in examples}
    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        enc = encode_pairs(tokenizer, [c[1] for c in chunk], [c[3] for c in chunk], max_len)
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        logit = model(**enc).float().cpu().tolist()
        for (eid, _, fid, _), sc in zip(chunk, logit):
            scores[eid].append((fid, sc))
    return {eid: sorted(v, key=lambda t: -t[1]) for eid, v in scores.items()}
