"""Which facts does the generator get to see?  (Pure Python, no torch.)

  topk_selector     the retriever's top-k facts            (normal inference)
  train_selector    top-k UNION gold facts                 (training; guarantees the
                    gold operands are copyable so every example gives signal)
  oracle_selector   gold facts, topped up with the best-ranked non-gold facts
                    to the same budget k                   (counterfactual run
                    used to attribute a failure to retrieval vs generation)

All selectors present facts in a fixed label-free order (see
grounding.ordered_fact_ids), so gold facts get no positional shortcut.

``ranked`` maps example id -> [(fact_id, score), ...] best first.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Sequence, Tuple

from .data import Example

Ranked = Dict[str, List[Tuple[str, float]]]


def released_ranked(examples: Sequence[Example]) -> Ranked:
    """Ranking from the retriever outputs shipped inside the FinQA json files."""
    return {e.id: list(e.released) for e in examples}


def topk_selector(ranked: Ranked, k: int) -> Callable[[Example], List[str]]:
    return lambda ex: [f for f, _ in ranked[ex.id][:k]]


def train_selector(ranked: Ranked, k: int) -> Callable[[Example], List[str]]:
    def sel(ex: Example) -> List[str]:
        return list(dict.fromkeys([f for f, _ in ranked[ex.id][:k]] + list(ex.gold_ids)))
    return sel


def oracle_selector(ranked: Ranked, k: int) -> Callable[[Example], List[str]]:
    def sel(ex: Example) -> List[str]:
        gold = list(ex.gold_ids)
        gs = set(gold)
        fill = [f for f, _ in ranked[ex.id] if f not in gs]
        return gold + fill[:max(0, k - len(gold))]
    return sel


def save_ranked(path: str, ranked: Ranked) -> None:
    with open(path, "w") as f:
        json.dump({k: [[fid, sc] for fid, sc in v] for k, v in ranked.items()}, f)


def load_ranked(path: str) -> Ranked:
    with open(path) as f:
        return {k: [(fid, float(sc)) for fid, sc in v] for k, v in json.load(f).items()}
