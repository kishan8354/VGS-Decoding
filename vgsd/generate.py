from __future__ import annotations

import copy
from typing import Optional

import torch
from transformers import LogitsProcessorList


def _greedy_config(model, max_new_tokens: int, pad_token_id: Optional[int]):
    cfg = copy.deepcopy(model.generation_config)
    cfg.do_sample = False
    cfg.num_beams = 1
    cfg.max_new_tokens = max_new_tokens
    cfg.max_length = None
    # neutralise sampling / penalty knobs some checkpoints ship with
    for k, v in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 50), ("repetition_penalty", 1.0)):
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if pad_token_id is not None:
        cfg.pad_token_id = pad_token_id
    return cfg


@torch.inference_mode()
def hf_generate(model, inputs: dict, max_new_tokens: int, processor=None,
                pad_token_id: Optional[int] = None, **gen_kwargs) -> torch.Tensor:
    """Greedy HF generate. Returns only the newly generated token ids, shape (rows, T)."""
    cfg = _greedy_config(model, max_new_tokens, pad_token_id)
    lp = LogitsProcessorList([processor]) if processor is not None else None
    out = model.generate(**inputs, generation_config=cfg, logits_processor=lp, use_cache=True,
                         **gen_kwargs)
    return out[:, inputs["input_ids"].shape[1]:]


@torch.inference_mode()
def stepwise_generate(model, inputs: dict, processor, max_new_tokens: int,
                      eos_token_ids, pad_token_id: int) -> torch.Tensor:
    """Cache-free fallback (O(T^2), fine for short VQA answers). Re-runs the full forward
    each step; use when a model's generate() does not play well with custom processors
    or batching. Must give the same tokens as hf_generate (see tests)."""
    inputs = dict(inputs)
    n = inputs["input_ids"].shape[0]
    device = inputs["input_ids"].device
    eos = torch.tensor(sorted(set(eos_token_ids)), device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    gen = []
    for _ in range(max_new_tokens):
        logits = model(**inputs, use_cache=False).logits[:, -1, :]
        scores = processor(inputs["input_ids"], logits) if processor is not None else logits
        nxt = scores.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_token_id), nxt)
        gen.append(nxt)
        done |= torch.isin(nxt, eos)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], nxt[:, None]], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(nxt[:, None])], dim=1)
        if "token_type_ids" in inputs:  # Gemma3: text tokens have type 0
            inputs["token_type_ids"] = torch.cat(
                [inputs["token_type_ids"], torch.zeros_like(nxt[:, None])], dim=1)
        if bool(done.all()):
            break
    return torch.stack(gen, dim=1)
