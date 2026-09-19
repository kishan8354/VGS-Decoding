"""Model adapters. Each adapter exposes:

    build_inputs(images: list[PIL], prompts: list[str]) -> dict of tensors on device
    generate(inputs, max_new_tokens, processor=None, pair_mode="batch") -> new token ids
    decode(ids) -> list[str]
    eos_token_ids, pad_token_id

For paired decoding, images/prompts are interleaved [orig_0, dist_0, orig_1, dist_1, ...]
and prompts are duplicated, so paired rows have identical length (no padding inside a pair).
"""
from __future__ import annotations

import os
import tempfile
import uuid

import torch
from PIL import Image

from .generate import hf_generate, stepwise_generate

DEFAULT_SYSTEM = "You are a medical image analysis expert."          # from VASE
DEFAULT_PROMPT = ("Answer this question as concisely as possible based on the provide images: "
                  "{question}")                                     # from VASE (typo kept on purpose)

MODEL_IDS = {
    "llava-med": "chaoyinshe/llava-med-v1.5-mistral-7b-hf",   # community HF conversion
    "llava-med-official": "microsoft/llava-med-v1.5-mistral-7b",
    "medgemma": "google/medgemma-4b-it",
    "chexagent": "StanfordAIMI/CheXagent-2-3b",
}


def _dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _eos_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _load_kwargs(dtype, device, load_in_4bit):
    kw = dict(torch_dtype=dtype, device_map=device)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    return kw


class _HFAdapter:
    model = None
    tokenizer = None
    supports_batch_padding = True

    def to_device(self, enc):
        out = {}
        for k, v in enc.items():
            if not torch.is_tensor(v):
                continue
            v = v.to(self.model.device)
            if k == "pixel_values":
                v = v.to(self.model.dtype)
            out[k] = v
        return out

    def generate(self, inputs, max_new_tokens=64, processor=None, pair_mode="batch", **gen_kwargs):
        if pair_mode == "batch" or processor is None:
            return hf_generate(self.model, inputs, max_new_tokens, processor, self.pad_token_id,
                               **gen_kwargs)
        return stepwise_generate(self.model, inputs, processor, max_new_tokens,
                                 self.eos_token_ids, self.pad_token_id)

    def decode(self, ids):
        return [s.strip() for s in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def expand2square(img: Image.Image, bg) -> Image.Image:
    """LLaVA-1.5 'pad' preprocessing (image_aspect_ratio='pad')."""
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new(img.mode, (s, s), bg)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


class LlavaMedHF(_HFAdapter):
    """LLaVA-Med v1.5 (Mistral-7B) via transformers' LlavaForConditionalGeneration."""

    def __init__(self, model_id=MODEL_IDS["llava-med"], dtype="bf16", device="cuda:0",
                 load_in_4bit=False, system_prompt=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, **_load_kwargs(_dtype(dtype), device, load_in_4bit)).eval()
        self.eos_token_ids = _eos_list(self.model.generation_config.eos_token_id) or [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
            else self.eos_token_ids[0]
        mean = getattr(self.processor.image_processor, "image_mean", [0.5, 0.5, 0.5])
        self.bg = tuple(int(255 * m) for m in mean)
        self.system_prompt = system_prompt  # LLaVA-Med mistral_instruct template has an empty system prompt

    def _format(self, prompt):
        """Use the checkpoint's chat template if it has one (image first, then text, as on the
        model card); otherwise LLaVA-Med's "mistral_instruct" format."""
        template = getattr(self.processor, "chat_template", None) or self.tokenizer.chat_template
        if template:
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
            try:
                return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            except Exception:
                return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        sys = f"{self.system_prompt}\n" if self.system_prompt else ""
        return f"[INST] {sys}<image>\n{prompt} [/INST]"

    def build_inputs(self, images, prompts):
        imgs = [expand2square(im, self.bg) for im in images]
        texts = [self._format(p) for p in prompts]
        enc = self.processor(images=imgs, text=texts, return_tensors="pt", padding=True)
        return self.to_device(enc)


class MedGemma(_HFAdapter):
    def __init__(self, model_id=MODEL_IDS["medgemma"], dtype="bf16", device="cuda:0",
                 load_in_4bit=False, system_prompt=DEFAULT_SYSTEM):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, **_load_kwargs(_dtype(dtype), device, load_in_4bit)).eval()
        self.eos_token_ids = _eos_list(self.model.generation_config.eos_token_id) or [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
            else self.eos_token_ids[0]
        self.system_prompt = system_prompt

    def build_inputs(self, images, prompts):
        convs = []
        for im, p in zip(images, prompts):
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            # VASE order: text first, then image
            msgs.append({"role": "user", "content": [{"type": "text", "text": p},
                                                     {"type": "image", "image": im}]})
            convs.append(msgs)
        enc = self.processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True)
        return self.to_device(enc)


class CheXagent2(_HFAdapter):
    """StanfordAIMI/CheXagent-2-3b (Qwen-VL style remote code: images are passed as file paths
    inside the prompt). Paired images are written to temp PNGs with equal-length names so the
    orig/dist rows tokenize to the same length. Use a separate env with the transformers version
    recommended on the model card."""

    supports_batch_padding = False

    def __init__(self, model_id=MODEL_IDS["chexagent"], dtype="bf16", device="cuda:0",
                 load_in_4bit=False, system_prompt="You are a helpful assistant."):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, **_load_kwargs(_dtype(dtype), device, load_in_4bit)).eval()
        gc = self.model.generation_config
        self.eos_token_ids = _eos_list(gc.eos_token_id) or _eos_list(self.tokenizer.eos_token_id)
        self.pad_token_id = gc.pad_token_id if gc.pad_token_id is not None else self.eos_token_ids[0]
        self.system_prompt = system_prompt
        self.tmpdir = tempfile.mkdtemp(prefix="chexagent_imgs_")

    def build_inputs(self, images, prompts):
        tag = uuid.uuid4().hex[:12]
        rows = []
        for j, (im, p) in enumerate(zip(images, prompts)):
            path = os.path.join(self.tmpdir, f"{tag}_{j:04d}.png")   # fixed-length names
            im.save(path)
            query = self.tokenizer.from_list_format([{"image": path}, {"text": p}])
            conv = [{"from": "system", "value": self.system_prompt}, {"from": "human", "value": query}]
            rows.append(self.tokenizer.apply_chat_template(conv, add_generation_prompt=True,
                                                           return_tensors="pt"))
        lens = {r.shape[1] for r in rows}
        if len(lens) != 1:
            raise RuntimeError("CheXagent adapter supports batch_size=1 (one orig/dist pair) only")
        ids = torch.cat(rows, 0).to(self.model.device)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids):
        out = []
        for row in ids.tolist():
            row = [t for t in row if t not in set(self.eos_token_ids) | {self.pad_token_id}]
            out.append(self.tokenizer.decode(row).strip())
        return out


class LlavaMedOfficial:
    """Official LLaVA-Med code path (pip install -e the LLaVA-Med repo, transformers 4.36).
    NOT tested in this repo's CI -- run `run_eval.py --method orig_pair` vs `--method greedy`
    first to confirm the pairing works in that environment."""

    supports_batch_padding = False

    def __init__(self, model_id=MODEL_IDS["llava-med-official"], dtype="fp16", device="cuda:0",
                 load_in_4bit=False, conv_mode="mistral_instruct", **_):
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_id, None, get_model_name_from_path(model_id), load_4bit=load_in_4bit, device=device)
        self.model.eval()
        self.conv_mode = conv_mode
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
            else self.tokenizer.eos_token_id

    def build_inputs(self, images, prompts):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        rows = []
        for p in prompts:
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + p)
            conv.append_message(conv.roles[1], None)
            rows.append(tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0))
        if len({r.shape[1] for r in rows}) != 1:
            raise RuntimeError("official LLaVA-Med adapter supports batch_size=1 pairs only")
        ids = torch.cat(rows, 0).to(self.model.device)
        pix = process_images(images, self.image_processor, self.model.config)
        return {"input_ids": ids, "images": pix.to(self.model.device, dtype=self.model.dtype)}

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens=64, processor=None, pair_mode="batch"):
        from transformers import LogitsProcessorList
        if pair_mode != "batch":
            raise NotImplementedError("stepwise mode not supported for the official LLaVA-Med code")
        lp = LogitsProcessorList([processor]) if processor is not None else None
        out = self.model.generate(inputs["input_ids"], images=inputs["images"], do_sample=False,
                                  num_beams=1, max_new_tokens=max_new_tokens, use_cache=True,
                                  logits_processor=lp, pad_token_id=self.pad_token_id)
        L = inputs["input_ids"].shape[1]
        # LLaVA's generate uses inputs_embeds, so it usually returns only new tokens
        if out.shape[1] > L and torch.equal(out[:, :L].clamp(min=0), inputs["input_ids"].clamp(min=0)):
            out = out[:, L:]
        return out

    def decode(self, ids):
        return [s.strip() for s in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def load_adapter(name: str, model_id: str | None = None, **kw):
    cls = {"llava-med": LlavaMedHF, "llava-med-official": LlavaMedOfficial,
           "medgemma": MedGemma, "chexagent": CheXagent2}[name]
    return cls(model_id=model_id or MODEL_IDS[name], **kw)
