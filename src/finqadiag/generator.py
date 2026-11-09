"""Program generator: encoder + pointer-style LSTM decoder (own implementation).

Input   question + the text of the selected facts (joined with the separator)
Output  a program token sequence  op ( a , b ) , op ( #0 , c ) ... EOF

Each output token is chosen from  [reserved vocabulary ; copy-candidates]:
  * reserved: operators, punctuation, EOF, constants, step references (#k), 'none'
  * copy-candidates: numbers / table-row labels found in the input facts, each
    represented by the mean of the encoder states over its subword span.
A softmax over the concatenation lets the model either generate a structural
token or point at an operand.  If a number occurs several times in the input,
every occurrence is a correct target and the loss marginalises over them.

The design follows the retriever-generator family of FinQA (encoder + LSTM
program decoder over a small DSL) but the implementation, the unified
reserved/copy softmax and the candidate extraction are written from scratch.
"""
from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Example
from .dsl import EOF, ProgramError, Step, parse_tokens
from .grounding import (RESERVED, RESERVED_IDX, Candidate, build_input,
                        gold_token_targets)

R = len(RESERVED)
SOS_IDX = RESERVED_IDX["SOS"]
EOF_IDX = RESERVED_IDX[EOF]


# --------------------------------------------------------------------------
# Encoding examples
# --------------------------------------------------------------------------
@dataclass
class GenItem:
    ex_id: str
    input_ids: List[int]
    spans: List[Tuple[int, int]]            # token span [s, e) per candidate
    cands: List[Candidate]                  # only candidates that survived truncation
    fact_ids: List[str]                     # facts placed in the input
    targets: Optional[List[Tuple[str, object]]] = None
    grounded: bool = True


def encode_item(tokenizer, ex: Example, fact_ids: Sequence[str], max_len: int = 512,
                with_targets: bool = True) -> GenItem:
    q, facts, cands = build_input(ex, fact_ids, sep=tokenizer.sep_token)
    enc = tokenizer(q, facts, truncation="only_second", max_length=max_len,
                    return_offsets_mapping=True)
    seq = enc.sequence_ids()
    offs = enc["offset_mapping"]
    idx = [i for i, s in enumerate(seq) if s == 1]
    starts = [offs[i][0] for i in idx]
    ends = [offs[i][1] for i in idx]
    kept, spans = [], []
    for c in cands:
        lo = bisect_right(ends, c.start)      # first token whose end > c.start
        hi = bisect_left(starts, c.end)       # tokens [lo, hi) start before c.end
        if lo < hi:
            kept.append(c)
            spans.append((idx[lo], idx[hi - 1] + 1))
    item = GenItem(ex.id, list(enc["input_ids"]), spans, kept, list(fact_ids))
    if with_targets:
        item.targets, item.grounded = gold_token_targets(ex.steps, kept)
    return item


def collate(items: Sequence[GenItem], pad_id: int) -> Dict[str, torch.Tensor]:
    B = len(items)
    L = max(len(it.input_ids) for it in items)
    C = max(1, max(len(it.cands) for it in items))
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros(B, L, dtype=torch.long)
    spans = torch.zeros(B, C, 2, dtype=torch.long)
    cmask = torch.zeros(B, C, dtype=torch.bool)
    for b, it in enumerate(items):
        n = len(it.input_ids)
        input_ids[b, :n] = torch.tensor(it.input_ids)
        attn[b, :n] = 1
        for j, (s, e) in enumerate(it.spans):
            spans[b, j, 0], spans[b, j, 1] = s, e
            cmask[b, j] = True
    batch = {"input_ids": input_ids, "attention_mask": attn, "spans": spans, "cmask": cmask}
    if items[0].targets is not None:
        T = max(len(it.targets) for it in items)
        allowed = torch.zeros(B, T, R + C, dtype=torch.bool)
        allowed[:, :, 0] = True                     # dummy slot keeps padded rows finite
        tmask = torch.zeros(B, T, dtype=torch.bool)
        prev_kind = torch.zeros(B, T, dtype=torch.long)
        prev_idx = torch.full((B, T), SOS_IDX, dtype=torch.long)
        for b, it in enumerate(items):
            prev = ("res", SOS_IDX)
            for t, (k, v) in enumerate(it.targets):
                prev_kind[b, t] = 0 if prev[0] == "res" else 1
                prev_idx[b, t] = prev[1] if prev[0] == "res" else prev[1][0]
                if k == "res":
                    allowed[b, t, :] = False
                    allowed[b, t, v] = True
                    tmask[b, t] = True
                elif k == "cand":
                    allowed[b, t, :] = False
                    for j in v:
                        allowed[b, t, R + j] = True
                    tmask[b, t] = True
                prev = (k, v) if k != "miss" else ("res", SOS_IDX)
        batch.update(allowed=allowed, tmask=tmask, prev_kind=prev_kind, prev_idx=prev_idx)
    return batch


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ProgramGenerator(nn.Module):
    def __init__(self, encoder: nn.Module, dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        H = encoder.config.hidden_size
        self.dim = dim
        self.enc_proj = nn.Linear(H, dim)
        self.res_emb = nn.Embedding(R, dim)
        self.init_h = nn.Linear(H, dim)
        self.init_c = nn.Linear(H, dim)
        self.cell = nn.LSTMCell(2 * dim, dim)
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.out_res = nn.Linear(2 * dim, R)
        self.out_cand = nn.Linear(2 * dim, dim)

    # ---- encoder side -----------------------------------------------------
    def encode(self, batch):
        h = self.encoder(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"]).last_hidden_state
        mem = self.enc_proj(h)                                   # [B,L,D]
        L = mem.size(1)
        pos = torch.arange(L, device=mem.device)[None, None, :]
        s, e = batch["spans"][..., 0:1], batch["spans"][..., 1:2]
        sm = ((pos >= s) & (pos < e) & batch["cmask"][..., None]).to(mem.dtype)   # [B,C,L]
        cand_emb = (sm @ mem) / sm.sum(-1, keepdim=True).clamp(min=1.0)           # [B,C,D]
        state = (torch.tanh(self.init_h(h[:, 0])), self.init_c(h[:, 0]))
        ctx = torch.zeros_like(state[0])
        return mem, cand_emb, state, ctx

    # ---- one decoder step ---------------------------------------------------
    def _embed_prev(self, kind, idx, cand_emb):
        res_e = self.res_emb(torch.where(kind == 0, idx, torch.zeros_like(idx)))
        cand_i = torch.where(kind == 1, idx, torch.zeros_like(idx))
        cand_e = cand_emb[torch.arange(cand_emb.size(0), device=cand_emb.device), cand_i]
        return torch.where((kind == 0)[:, None], res_e, cand_e)

    def step(self, prev_emb, state, ctx, mem, amask, cand_emb, cmask):
        h, c = self.cell(torch.cat([prev_emb, ctx], -1), state)
        scores = (mem @ self.attn_q(h).unsqueeze(-1)).squeeze(-1).float()
        scores = scores.masked_fill(~amask, -1e9)
        a = torch.softmax(scores, -1).to(mem.dtype)
        ctx = (a.unsqueeze(1) @ mem).squeeze(1)
        feat = self.drop(torch.cat([h, ctx], -1))
        res_logits = self.out_res(feat).float()
        q = self.out_cand(feat)
        cand_logits = ((cand_emb @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.dim)).float()
        cand_logits = cand_logits.masked_fill(~cmask, -1e9)
        return torch.cat([res_logits, cand_logits], -1), (h, c), ctx

    # ---- training ------------------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        mem, cand_emb, state, ctx = self.encode(batch)
        amask = batch["attention_mask"].bool()
        T = batch["prev_kind"].size(1)
        total, count = mem.new_zeros(()).float(), 0
        for t in range(T):
            prev = self._embed_prev(batch["prev_kind"][:, t], batch["prev_idx"][:, t], cand_emb)
            logits, state, ctx = self.step(prev, state, ctx, mem, amask, cand_emb, batch["cmask"])
            lp = F.log_softmax(logits, -1).masked_fill(~batch["allowed"][:, t], float("-inf"))
            tgt = torch.logsumexp(lp, -1)
            m = batch["tmask"][:, t]
            if m.any():
                total = total - tgt[m].sum()
                count += int(m.sum())
        return total / max(count, 1)

    # ---- inference -------------------------------------------------------------
    @torch.no_grad()
    def generate(self, batch, max_len: int = 48) -> List[List[int]]:
        """Greedy decoding. Returns, per example, indices into [reserved ; candidates]."""
        mem, cand_emb, state, ctx = self.encode(batch)
        amask = batch["attention_mask"].bool()
        B = mem.size(0)
        kind = torch.zeros(B, dtype=torch.long, device=mem.device)
        idx = torch.full((B,), SOS_IDX, dtype=torch.long, device=mem.device)
        outs: List[List[int]] = [[] for _ in range(B)]
        done = [False] * B
        for _ in range(max_len):
            prev = self._embed_prev(kind, idx, cand_emb)
            logits, state, ctx = self.step(prev, state, ctx, mem, amask, cand_emb, batch["cmask"])
            logits[:, SOS_IDX] = float("-inf")
            pred = logits.argmax(-1)
            for b in range(B):
                if not done[b]:
                    outs[b].append(int(pred[b]))
                    done[b] = int(pred[b]) == EOF_IDX
            if all(done):
                break
            is_c = pred >= R
            kind = is_c.long()
            idx = torch.where(is_c, pred - R, pred)
        return outs


    @torch.no_grad()
    def generate_beam(self, batch, beam: int = 8, max_len: int = 48
                      ) -> List[List[Tuple[List[int], float]]]:
        """Beam search. Returns, per example, up to ``beam`` finished hypotheses
        ``(indices into [reserved ; candidates], sum of log-probabilities)``, best first.

        Hypotheses of one example are decoded as a batch of rows sharing the encoder
        output.  A hypothesis finishes when it emits EOF.  Search stops once ``beam``
        hypotheses are finished and no active one can still beat the worst of them
        (sums of log-probabilities only decrease).
        """
        mem, cand_emb, state, ctx = self.encode(batch)
        amask = batch["attention_mask"].bool()
        dev = mem.device
        results: List[List[Tuple[List[int], float]]] = []
        for b in range(mem.size(0)):
            m, ce = mem[b:b + 1], cand_emb[b:b + 1]
            am, cm = amask[b:b + 1], batch["cmask"][b:b + 1]
            h, c, cx = state[0][b:b + 1], state[1][b:b + 1], ctx[b:b + 1]
            kind = torch.zeros(1, dtype=torch.long, device=dev)
            idx = torch.full((1,), SOS_IDX, dtype=torch.long, device=dev)
            seqs: List[List[int]] = [[]]
            scores = torch.zeros(1, device=dev)
            finished: List[Tuple[List[int], float]] = []
            for _ in range(max_len):
                K = len(seqs)
                prev = self._embed_prev(kind, idx, ce.expand(K, -1, -1))
                logits, (h, c), cx = self.step(prev, (h, c), cx, m.expand(K, -1, -1), am.expand(K, -1),
                                               ce.expand(K, -1, -1), cm.expand(K, -1))
                logits[:, SOS_IDX] = float("-inf")
                lp = F.log_softmax(logits, -1)
                V = lp.size(-1)
                tot = (scores[:, None] + lp).reshape(-1)
                top = tot.topk(min(2 * beam, tot.numel()))
                keep_rows, keep_tok, keep_sc, new_seqs = [], [], [], []
                for sc, flat in zip(top.values.tolist(), top.indices.tolist()):
                    if sc < -1e8:                       # masked (padding candidate) slot
                        continue
                    row, tok = divmod(flat, V)
                    if tok == EOF_IDX:
                        finished.append((seqs[row] + [tok], sc))
                    elif len(keep_rows) < beam:
                        keep_rows.append(row); keep_tok.append(tok); keep_sc.append(sc)
                        new_seqs.append(seqs[row] + [tok])
                if not keep_rows:
                    break
                finished.sort(key=lambda x: -x[1])
                finished = finished[:beam]
                if len(finished) >= beam and max(keep_sc) < finished[-1][1]:
                    break
                rows = torch.tensor(keep_rows, device=dev)
                h, c, cx = h[rows], c[rows], cx[rows]
                tokt = torch.tensor(keep_tok, device=dev)
                is_c = tokt >= R
                kind = is_c.long()
                idx = torch.where(is_c, tokt - R, tokt)
                scores = torch.tensor(keep_sc, device=dev)
                seqs = new_seqs
            finished.sort(key=lambda x: -x[1])
            results.append(finished[:beam])
        return results


def indices_to_tokens(out: Sequence[int], cands: Sequence[Candidate]) -> List[str]:
    toks = []
    for i in out:
        if i < R:
            toks.append(RESERVED[i])
        else:
            j = i - R
            toks.append(cands[j].text if j < len(cands) else "<bad>")
    return toks


def tokens_to_steps(tokens: Sequence[str]) -> Optional[List[Step]]:
    try:
        return parse_tokens(tokens)
    except ProgramError:
        return None


# --------------------------------------------------------------------------
# High-level helpers
# --------------------------------------------------------------------------
FactSelector = Callable[[Example], Sequence[str]]


def build_items(tokenizer, examples: Sequence[Example], selector: FactSelector,
                max_len: int = 512, with_targets: bool = True,
                drop_ungrounded: bool = False) -> List[GenItem]:
    items = []
    for ex in examples:
        it = encode_item(tokenizer, ex, selector(ex), max_len, with_targets)
        if drop_ungrounded and not it.grounded:
            continue
        items.append(it)
    return items


@torch.no_grad()
def predict(model: ProgramGenerator, tokenizer, items: Sequence[GenItem], device,
            batch_size: int = 16, max_len: int = 48) -> Dict[str, Dict[str, object]]:
    """Greedy-decode every item. Returns {id: {"tokens": [...], "facts": [...]}}."""
    model.eval()
    out: Dict[str, Dict[str, object]] = {}
    pad = tokenizer.pad_token_id
    order = sorted(range(len(items)), key=lambda i: len(items[i].input_ids))
    for s in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[s:s + batch_size]]
        batch = {k: v.to(device) for k, v in collate(chunk, pad).items()}
        for it, o in zip(chunk, model.generate(batch, max_len)):
            out[it.ex_id] = {"tokens": indices_to_tokens(o, it.cands), "facts": it.fact_ids}
    return out


@torch.no_grad()
def predict_beam(model: ProgramGenerator, tokenizer, items: Sequence[GenItem], device,
                 beam: int = 8, batch_size: int = 8, max_len: int = 48
                 ) -> Dict[str, Dict[str, object]]:
    """Beam-decode every item. Returns {id: {"beams": [{"tokens": [...], "logp": float}], "facts": [...]}}."""
    model.eval()
    out: Dict[str, Dict[str, object]] = {}
    pad = tokenizer.pad_token_id
    order = sorted(range(len(items)), key=lambda i: len(items[i].input_ids))
    for s in range(0, len(order), batch_size):
        chunk = [items[i] for i in order[s:s + batch_size]]
        batch = {k: v.to(device) for k, v in collate(chunk, pad).items()}
        for it, hyps in zip(chunk, model.generate_beam(batch, beam, max_len)):
            out[it.ex_id] = {"beams": [{"tokens": indices_to_tokens(o, it.cands), "logp": lp}
                                       for o, lp in hyps],
                             "facts": it.fact_ids}
    return out
