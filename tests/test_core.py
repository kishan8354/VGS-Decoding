"""CPU tests with tiny randomly-initialised models (no downloads). Run: python -m pytest -q tests"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vgsd.generate import hf_generate, stepwise_generate  # noqa: E402
from vgsd.metrics import closed_correct, mcnemar_exact, open_recall, summarize  # noqa: E402
from vgsd.noise import distort, prepare_pair  # noqa: E402
from vgsd.vgs import PairedDecodingProcessor, vcd_contrast, vgs_from_logprobs, vgs_reweight  # noqa: E402

EOS, PAD, IMG = 2, 0, 100


# ---------------------------------------------------------------- VGS math
def test_vgs_matches_paper_formula():
    torch.manual_seed(0)
    lo, ld = torch.randn(4, 50), torch.randn(4, 50)
    po, pd = lo.softmax(-1), ld.softmax(-1)
    direct = (po - pd) / (po + pd)
    stable = vgs_from_logprobs(lo.log_softmax(-1), ld.log_softmax(-1))
    assert torch.allclose(direct, stable, atol=1e-5)
    assert stable.abs().max() <= 1


def test_vgs_underflow_and_zero_probs():
    lo = torch.tensor([[0.0, -1e4, float("-inf"), -3.0]])
    ld = torch.tensor([[0.0, -2e4, float("-inf"), float("-inf")]])
    v = vgs_from_logprobs(lo.log_softmax(-1), ld.log_softmax(-1))
    assert torch.isfinite(v).all()
    assert v[0, 2] == 0 and v[0, 3] == 1 and v[0, 1] == 1


def test_reweight_eq5_and_alpha0():
    torch.manual_seed(1)
    lo, ld = torch.randn(3, 40), torch.randn(3, 40)
    lf, vgs, lpo, _ = vgs_reweight(lo, ld, alpha=1.5, delta=0.01)
    po = lpo.exp()
    ref = po * torch.clamp(1 + 1.5 * vgs, min=0.01)
    ref = ref / ref.sum(-1, keepdim=True)
    assert torch.allclose(lf.exp(), ref, atol=1e-6)
    lf0, *_ = vgs_reweight(lo, ld, alpha=0.0)
    assert torch.allclose(lf0, lo.log_softmax(-1), atol=1e-6)


def test_vcd_formula():
    lo = torch.tensor([[2.0, 1.9, -5.0]]); ld = torch.tensor([[2.5, 0.0, -5.0]])
    out = vcd_contrast(lo, ld, alpha=1.0, beta=0.1)
    assert torch.isinf(out[0, 2])                 # below plausibility cut-off
    assert out[0].argmax() == 1                   # 2*1.9-0 > 2*2-2.5


# ---------------------------------------------------------------- tiny models
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                      eos_token_id=EOS, pad_token_id=PAD, bos_token_id=1)
    m = LlamaForCausalLM(cfg).eval()
    m.generation_config.eos_token_id, m.generation_config.pad_token_id = EOS, PAD
    return m


def tiny_llava():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
    torch.manual_seed(0)
    vc = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                          image_size=28, patch_size=14, projection_dim=32)
    tc = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                     eos_token_id=EOS, pad_token_id=PAD, bos_token_id=1)
    cfg = LlavaConfig(vision_config=vc, text_config=tc, image_token_index=IMG,
                      vision_feature_select_strategy="default", vision_feature_layer=-1)
    m = LlavaForConditionalGeneration(cfg).eval()
    m.generation_config.eos_token_id, m.generation_config.pad_token_id = EOS, PAD
    return m


def llava_pair_inputs(n_questions=1):
    torch.manual_seed(123)
    ids, pix = [], []
    for q in range(n_questions):
        row = torch.tensor([1, IMG, IMG, IMG, IMG, 10 + q, 11, 12 + q])
        img = torch.randn(3, 28, 28)
        ids += [row, row]
        pix += [img, img + 0.8 * torch.randn(3, 28, 28)]   # orig, distorted
    ids = torch.stack(ids)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.stack(pix)}


def _gen_pair(model, inputs, method, alpha, mode):
    proc = PairedDecodingProcessor(method=method, alpha=alpha, eos_token_ids=[EOS], trace_topk=3)
    if mode == "batch":
        out = hf_generate(model, inputs, 12, proc, PAD)
    else:
        out = stepwise_generate(model, inputs, proc, 12, [EOS], PAD)
    return out, proc


def test_pairs_stay_in_lockstep_and_match_stepwise():
    model = tiny_llava()
    inputs = llava_pair_inputs(2)
    for method, alpha in [("vgs", 1.0), ("vgs", 3.0), ("vcd", 1.0)]:
        b, proc = _gen_pair(model, inputs, method, alpha, "batch")
        s, _ = _gen_pair(model, inputs, method, alpha, "stepwise")
        assert torch.equal(b[0::2], b[1::2]), "orig/dist rows diverged"
        T = min(b.shape[1], s.shape[1])
        assert torch.equal(b[:, :T], s[:, :T]), f"batch != stepwise for {method}"
        assert len(proc.trace[0]) == proc.steps[0] and proc.steps[0] > 0


def test_orig_pair_equals_plain_greedy():
    model = tiny_llava()
    inputs = llava_pair_inputs(2)
    pair_out, _ = _gen_pair(model, inputs, "orig", 0.0, "batch")
    vgs0_out, _ = _gen_pair(model, inputs, "vgs", 0.0, "batch")
    single = {k: v[0::2] for k, v in inputs.items()}
    greedy = hf_generate(model, single, 12, None, PAD)
    T = greedy.shape[1]
    assert torch.equal(pair_out[0::2, :T], greedy)
    assert torch.equal(vgs0_out[0::2, :T], greedy)


def test_vgs_changes_something_at_high_alpha():
    model = tiny_llava()
    inputs = llava_pair_inputs(2)
    _, proc = _gen_pair(model, inputs, "vgs", 5.0, "batch")
    assert sum(proc.flips) > 0


def test_text_only_model_pairing():
    model = tiny_llama()
    ids = torch.tensor([[1, 5, 6, 7], [1, 5, 9, 7]])   # "distorted" = perturbed prompt
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    b, _ = _gen_pair(model, inputs, "vgs", 1.0, "batch")
    s, _ = _gen_pair(model, inputs, "vgs", 1.0, "stepwise")
    T = min(b.shape[1], s.shape[1])
    assert torch.equal(b[0], b[1]) and torch.equal(b[:, :T], s[:, :T])


def test_gemma3_medgemma_architecture():
    """Same checks on a tiny Gemma3ForConditionalGeneration (MedGemma's class; needs torch>=2.6)."""
    import pytest
    if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
        pytest.skip("Gemma3 masks need torch>=2.6")
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig
    torch.manual_seed(0)
    tc = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                          sliding_window=16, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=1)
    vc = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                            image_size=28, patch_size=7)
    m = Gemma3ForConditionalGeneration(Gemma3Config(text_config=tc, vision_config=vc, mm_tokens_per_image=4,
                                                    image_token_index=IMG, boi_token_index=101,
                                                    eoi_token_index=102)).eval()
    m.generation_config.eos_token_id, m.generation_config.pad_token_id = EOS, PAD
    torch.nn.init.normal_(m.model.multi_modal_projector.mm_input_projection_weight, std=0.5)  # zero-init otherwise

    class NoImgTok(PairedDecodingProcessor):   # random model must not emit image placeholders
        def __call__(self, ids, scores):
            scores = scores.clone(); scores[:, IMG:IMG + 3] = -1e9
            return super().__call__(ids, scores)

    rows, pix = [], []
    for q in range(2):
        r = torch.tensor([1, 20 + q, 101, IMG, IMG, IMG, IMG, 102, 30, 31 + q]); img = torch.randn(3, 28, 28)
        rows += [r, r]; pix += [img, img + 0.8 * torch.randn(3, 28, 28)]
    ids = torch.stack(rows)
    inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": torch.stack(pix),
           "token_type_ids": (ids == IMG).long()}
    total_flips = 0
    for method, alpha in [("vgs", 1.0), ("vgs", 3.0), ("vcd", 1.0)]:
        pb = NoImgTok(method=method, alpha=alpha, eos_token_ids=[EOS])
        b = hf_generate(m, inp, 12, pb, PAD)
        s = stepwise_generate(m, inp, NoImgTok(method=method, alpha=alpha, eos_token_ids=[EOS]), 12, [EOS], PAD)
        T = min(b.shape[1], s.shape[1])
        assert torch.equal(b[0::2], b[1::2]) and torch.equal(b[:, :T], s[:, :T])
        total_flips += sum(pb.flips)
    assert total_flips > 0


# ---------------------------------------------------------------- noise
def test_noise_deterministic_and_matches_vase_recipe():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (40, 60), dtype=np.uint8)).convert("L")
    o1, d1 = prepare_pair(img, "q1", size=32)
    o2, d2 = prepare_pair(img, "q1", size=32)
    _, d3 = prepare_pair(img, "q2", size=32)
    assert o1.size == d1.size == (32, 32) and d1.mode == "RGB"
    assert np.array_equal(np.array(d1), np.array(d2))
    assert not np.array_equal(np.array(d1), np.array(d3))
    # reference re-implementation of VASE AddGaussianNoise -> AddPoissonNoise -> ToPILImage
    g = torch.Generator().manual_seed(7)
    x = torch.from_numpy(np.asarray(o1, dtype=np.float32) / 255)
    x = (x + torch.randn(x.shape, generator=g) * 0.07).clamp(0, 1)
    x = (torch.poisson((x * 70).clamp(0, 255), generator=g) / 70).clamp(0, 1)
    ref = (x * 255).to(torch.uint8).numpy()
    assert np.array_equal(np.array(distort(o1, 0.07, 70, seed=7)), ref)
    diff = np.abs(np.array(d1, dtype=float) - np.array(o1, dtype=float)).mean()
    assert 5 < diff < 80


# ---------------------------------------------------------------- DoLA
def test_dola_runs_and_changes_output():
    from transformers import LlamaConfig, LlamaForCausalLM

    from vgsd.dola import DoLaProcessor, default_candidate_layers, find_decoder_layers
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                      eos_token_id=EOS, pad_token_id=PAD, bos_token_id=1)
    m = LlamaForCausalLM(cfg).eval()
    m.generation_config.eos_token_id, m.generation_config.pad_token_id = EOS, PAD
    assert default_candidate_layers(8) == [4, 6]
    ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
    inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    base = hf_generate(m, inp, 12, None, PAD)
    with DoLaProcessor(m, repetition_penalty=1.2) as p:
        out = hf_generate(m, inp, 12, p, PAD)
        assert len(p._handles) == len(p.candidates)
    assert len(p._handles) == 0                      # hooks removed on exit
    assert p.chosen_layers and set(p.chosen_layers) <= set(p.candidates)
    assert base[0].tolist() != out[0].tolist()       # DoLA actually changes the output
    assert find_decoder_layers(m) is m.model.layers


def test_dola_finds_layers_on_vlm():
    from vgsd.dola import find_decoder_layers
    m = tiny_llava()
    assert len(find_decoder_layers(m)) == 2          # language model, not the vision tower


# ---------------------------------------------------------------- metrics
def test_metrics():
    assert closed_correct("No, there is no effusion.", "no") == 1
    assert closed_correct("Normal study", "no", "word") == 0
    assert closed_correct("Normal study", "no", "substring") == 1
    assert closed_correct("Yes. No mass seen", "no", "first") == 0
    assert open_recall("Right lower lobe", "right lung") == 0.5
    recs = ([{"pred": "yes", "answer": "yes", "is_closed": True}] * 3 +
            [{"pred": "left", "answer": "left lung", "is_closed": False}] * 2)
    s = summarize(recs)
    assert abs(s["closed"] - 100) < 1e-9 and abs(s["open"] - 50) < 1e-9 and abs(s["overall"] - 80) < 1e-9
    # paper's Overall column is the sample-weighted mean
    assert abs((200 * 34.45 + 251 * 68.92) / 451 - 53.64) < 0.02
    assert mcnemar_exact([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
