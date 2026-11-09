"""Training / evaluation loops for the retriever and the generator (own code).

Both loops: AdamW, linear warm-up + decay, gradient clipping, optional fp16 on
CUDA, best-checkpoint selection on the dev set.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import torch

from .data import Example
from .generator import (GenItem, ProgramGenerator, build_items, collate, predict,
                        tokens_to_steps)
from .metrics import execution_correct, program_correct
from .retriever import FactScorer, encode_pairs, make_pairs, rank_facts
from .selectors import Ranked, save_ranked
from .taxonomy import retrieval_breakdown


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(name: str):
    """HuggingFace encoder + fast tokenizer (needs internet / cached weights)."""
    from transformers import AutoModel, AutoTokenizer
    return AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name, use_fast=True)


def _scheduler(opt, total_steps: int, warmup_frac: float = 0.1):
    warm = max(1, int(total_steps * warmup_frac))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _amp(device, fp16: bool):
    on = fp16 and device.type == "cuda"
    return on, torch.amp.GradScaler("cuda", enabled=on)


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------
def evaluate_retriever(model, tokenizer, examples, device, k: int = 5, batch_size: int = 64):
    ranked = rank_facts(model, tokenizer, examples, device, batch_size)
    r = retrieval_breakdown(examples, {e: [f for f, _ in v] for e, v in ranked.items()},
                            "all", k=k)["all"]
    return ranked, r


def train_retriever(model: FactScorer, tokenizer, train_ex: Sequence[Example],
                    dev_ex: Sequence[Example], out_dir: str, epochs: int = 3,
                    lr: float = 2e-5, batch_size: int = 32, neg_ratio: float = 3.0,
                    max_len: int = 256, fp16: bool = True, seed: int = 0, log_every: int = 50):
    device = get_device()
    set_seed(seed)
    model.to(device)
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = None
    rng = random.Random(seed)
    use_amp, scaler = _amp(device, fp16)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, history, step = -1.0, [], 0
    sched = None
    for epoch in range(epochs):
        pairs = make_pairs(train_ex, neg_ratio, rng)
        if sched is None:
            steps_per_epoch = (len(pairs) + batch_size - 1) // batch_size
            sched = _scheduler(opt, steps_per_epoch * epochs)
        model.train()
        t0 = time.time()
        run = 0.0
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s:s + batch_size]
            enc = encode_pairs(tokenizer, [train_ex[i].question for i, _, _ in chunk],
                               [train_ex[i].fact_text(f) for i, f, _ in chunk], max_len)
            y = torch.tensor([l for _, _, l in chunk], dtype=torch.float, device=device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logit = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += loss.item()
            if step % log_every == 0:
                print(f"[retriever] epoch {epoch} step {step} loss {run / log_every:.4f}", flush=True)
                run = 0.0
        ranked, r = evaluate_retriever(model, tokenizer, dev_ex, device)
        print(f"[retriever] epoch {epoch} done in {time.time() - t0:.0f}s | dev top-5 full recall "
              f"{r['full_recall']:.4f} fact recall {r['fact_recall']:.4f}", flush=True)
        history.append({"epoch": epoch, **r})
        if r["full_recall"] > best:
            best = r["full_recall"]
            torch.save(model.state_dict(), os.path.join(out_dir, "retriever.pt"))
    json.dump(history, open(os.path.join(out_dir, "retriever_history.json"), "w"), indent=2)
    return best


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
def evaluate_generator(preds: Dict[str, Dict[str, object]], examples: Sequence[Example]):
    ea = pa = 0
    for ex in examples:
        steps = tokens_to_steps(preds[ex.id]["tokens"])
        ea += execution_correct(steps, ex.table, ex.exe_ans)
        pa += program_correct(steps, ex.program)
    return ea / len(examples), pa / len(examples)


def train_generator(model: ProgramGenerator, tokenizer, train_items: Sequence[GenItem],
                    dev_ex: Sequence[Example], dev_items: Sequence[GenItem], out_dir: str,
                    epochs: int = 15, lr_enc: float = 2e-5, lr_dec: float = 1e-3,
                    batch_size: int = 8, grad_accum: int = 2, fp16: bool = True,
                    seed: int = 0, eval_every: int = 1, log_every: int = 50):
    device = get_device()
    set_seed(seed)
    model.to(device)
    os.makedirs(out_dir, exist_ok=True)
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr_enc},
                             {"params": dec_params, "lr": lr_dec}], weight_decay=0.01)
    steps_total = epochs * ((len(train_items) + batch_size * grad_accum - 1) // (batch_size * grad_accum))
    sched = _scheduler(opt, steps_total)
    use_amp, scaler = _amp(device, fp16)
    pad = tokenizer.pad_token_id
    rng = random.Random(seed)
    best, history, step = -1.0, [], 0
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_items)))
        rng.shuffle(order)
        t0, run, n_run = time.time(), 0.0, 0
        for bi, s in enumerate(range(0, len(order), batch_size)):
            batch = collate([train_items[i] for i in order[s:s + batch_size]], pad)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(batch)
            scaler.scale(loss / grad_accum).backward()
            run += loss.item()
            n_run += 1
            if (bi + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    print(f"[generator] epoch {epoch} step {step}/{steps_total} "
                          f"loss {run / n_run:.4f}", flush=True)
                    run, n_run = 0.0, 0
        rec = {"epoch": epoch, "seconds": round(time.time() - t0)}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            preds = predict(model, tokenizer, dev_items, device)
            ea, pa = evaluate_generator(preds, dev_ex)
            rec.update(dev_ea=ea, dev_pa=pa)
            print(f"[generator] epoch {epoch} | dev EA {ea:.4f} PA {pa:.4f} ({rec['seconds']}s)", flush=True)
            if ea > best:
                best = ea
                torch.save(model.state_dict(), os.path.join(out_dir, "generator.pt"))
        history.append(rec)
        json.dump(history, open(os.path.join(out_dir, "generator_history.json"), "w"), indent=2)
    return best
