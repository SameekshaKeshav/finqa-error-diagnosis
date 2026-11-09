"""Tiny offline tokenizer + encoder for CPU smoke tests (no pretrained weights).

HuggingFace may be unreachable (it is in the sandbox I develop in), and real
runs need a GPU anyway.  These helpers build a word-level fast tokenizer from
the FinQA text itself and a 2-layer random RoBERTa, so the full
retriever/generator code path (offsets -> candidate spans -> pointer decoding
-> training -> greedy decoding) can be tested on CPU.  They are test tools;
they are never used for reported results.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .data import Example


def build_tiny_tokenizer(examples: Sequence[Example], max_vocab: int = 6000):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    cnt: Counter = Counter()
    pre = pre_tokenizers.Whitespace()
    for ex in examples:
        for text in [ex.question] + list(ex.facts().values()):
            cnt.update(w for w, _ in pre.pre_tokenize_str(text))
    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]          # ids 0..4 (roberta layout)
    vocab = {t: i for i, t in enumerate(specials)}
    for w, _ in cnt.most_common(max_vocab - len(specials)):
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre
    tk.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token="<s>", eos_token="</s>", sep_token="</s>",
        cls_token="<s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")


def build_tiny_encoder(vocab_size: int, hidden: int = 64, layers: int = 2, heads: int = 4):
    from transformers import RobertaConfig, RobertaModel
    cfg = RobertaConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers,
                        num_attention_heads=heads, intermediate_size=hidden * 2,
                        max_position_embeddings=514, pad_token_id=1, type_vocab_size=1,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    return RobertaModel(cfg)
