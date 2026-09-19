# VGS-Decoding: independent reproduction

Reproduction of *VGS-Decoding: Visual Grounding Score Guided Decoding for Hallucination Mitigation in Medical VLMs* (Kolli et al., arXiv 2603.20314). The authors have not released code, so this is a clean re-implementation. Details the paper leaves open were filled in from the VASE codebase (github.com/Merrical/VASE, MICCAI 2025, the paper's ref. [19]). VASE uses the same noise values, the same datasets and the same models.

## What is here

```
vgsd/vgs.py        VGS (Eq. 4), reweighting (Eq. 5), VCD baseline, paired LogitsProcessor
vgsd/noise.py      Gaussian(σ=0.07) → Poisson(scale=70) distortion, exactly as in VASE
vgsd/models.py     adapters: llava-med (HF), llava-med-official, medgemma, chexagent
vgsd/data.py       VQA-RAD (HF flaviagiammarino/vqa-rad), MIMIC-Diff-VQA (PhysioNet)
vgsd/metrics.py    closed acc, open token recall, weighted overall, McNemar, paired bootstrap
scripts/run_eval.py         resumable JSONL runs (greedy | vgs | vcd | orig_pair)
scripts/score.py            paper-style table, Δ, flip rate, sec/question, p-values
scripts/reproduce_vqarad.sh Table 1 (VQA-RAD) + Table 3 α-sweep for one model
tests/test_core.py          11 CPU tests on tiny random LLaVA / Gemma3 / Llama models
```

**How the decoding is implemented.** Each question is placed in the batch twice: row 2i gets the original image and row 2i+1 gets the distorted image. At every step the processor computes P_final from the pair and writes it into both rows. Greedy decoding therefore picks the same token in both rows, so the distorted branch always sees the same prefix y<t, as in Algorithm 1. HF `generate` still manages the KV caches. The result is one batched forward pass per step, not two sequential passes. VGS is computed as `tanh((log P_o − log P_d)/2)`, which equals (P_o−P_d)/(P_o+P_d) exactly but does not underflow over large vocabularies.

**Verified on CPU with tiny random models** (LLaVA and Gemma3 architectures):
- The orig and dist rows stay in lock-step.
- Batched generation matches a cache-free stepwise re-implementation token for token, for VGS at α=1 and α=3 and for VCD.
- `orig_pair` and VGS with α=0 give exactly the plain greedy output.
- The noise matches the VASE recipe exactly.
- The metrics reproduce the paper's Overall column (e.g. (200·34.45 + 251·68.92)/451 = 53.64).

**Not tested here:** real checkpoints (no GPU in the build sandbox), the CheXagent remote code, and the official LLaVA-Med code path. The first run on each model must be the sanity check below.

## Setup

```bash
conda create -n vgs python=3.10 -y && conda activate vgs
pip install -r requirements.txt
huggingface-cli login          # MedGemma is gated: accept the licence on its HF page first
python -m pytest -q tests      # should print 11 passed
```

CheXagent-2-3b ships remote code written for an older transformers release. Put it in its own env and install the transformers version given on the model card. The official LLaVA-Med code (`--model llava-med-official`) also needs its own env: install the LLaVA-Med repo with `pip install -e .`, which pins transformers 4.36.

GPU memory at bf16 with one orig/dist pair: MedGemma-4B needs about 10 GB, CheXagent-3B about 8 GB and LLaVA-Med-7B about 16 GB. `--load_in_4bit` works but changes the numbers.

## Run order (do not skip step 1)

```bash
# 1. sanity: pairing must not change greedy output on the real model
python scripts/run_eval.py --model medgemma --dataset vqa-rad --method greedy    --limit 50 --out runs/s/greedy.jsonl
python scripts/run_eval.py --model medgemma --dataset vqa-rad --method orig_pair --limit 50 --out runs/s/orig_pair.jsonl
python scripts/score.py --agreement runs/s/greedy.jsonl runs/s/orig_pair.jsonl   # expect ~100% (rare bf16 near-tie differences); use --dtype fp32 to confirm

# 2. full VQA-RAD for one model (greedy, VCD, VGS α∈{0.5,1,1.5,2}, trace file)
bash scripts/reproduce_vqarad.sh medgemma
```

If agreement is low, try `--pair_mode stepwise`. It is slower but does not depend on the model's `generate` implementation.

## Targets (paper, Overall %)

| Model | Data | Greedy | VCD | VGS α=1 |
|---|---|---|---|---|
| LLaVA-Med | VQA-RAD | 53.64 | 47.71 | 57.75 |
| CheXagent | VQA-RAD | 49.24 | 47.78 | 50.10 |
| MedGemma | VQA-RAD | 56.32 | 54.45 | 64.48 |
| LLaVA-Med | MIMIC-Diff | 35.39 | 33.33 | 39.86 |
| CheXagent | MIMIC-Diff | 57.79 | 53.42 | 57.91 |
| MedGemma | MIMIC-Diff | 43.16 | 43.61 | 52.28 |

Table 3 α-sweep (VQA-RAD Overall): LLaVA-Med 56.95 / 57.75 / 58.79 / 57.50 and MedGemma 59.00 / 64.48 / 64.37 / 65.90, for α = 0.5 / 1.0 / 1.5 / 2.0.

Judge success by whether the **greedy baseline** lands close to the paper's number first. Absolute values depend on prompt and matching rules. The Δ from VGS is the result that has to replicate.

## Unspecified details (choices made; worth emailing the authors)

1. **Noise.** Eq. 2 says `V + N + Poisson(λ)`. Taken literally that would saturate a [0,1] image, so the code follows VASE: clamp(x+N(0,0.07²)), then Poisson(70x)/70, per RGB channel, after a 512×512 resize. Use `--resize 0` to skip the resize and `--noise_mode vase_full` to add VASE's geometric jitter.
2. **Prompt.** Default is VASE's "Answer this question as concisely as possible based on the provide images: {q}", with system prompt "You are a medical image analysis expert." for MedGemma. Override with `--prompt`.
3. **Closed-ended matching.** LLaVA-Med's lenient substring test counts "no" as present in "normal". Scores are reported under both `word` and `substring` rules; check which one reproduces the greedy baseline.
4. **VQA-RAD split.** Closed is defined as a yes/no answer. The loader prints the open/closed counts next to the paper's 200/251.
5. **MIMIC-Diff-VQA.** The paper gives 13,121 test questions (8,380 open, 4,741 closed) but not how they were selected. Difference questions need two images, and it is unclear how single-image LLaVA-Med was fed. Access requires PhysioNet credentialing (CITI course), so start that application now.
6. **LLaVA-Med checkpoint.** The HF conversion uses the LLaVA "pad" preprocessing here. The paper may have used the official code, which is available as `llava-med-official`.
7. **Table 2.** The "Fixed weight (VCD-style)" row is identical to VCD in Table 1. What "VGS metric only" means is not defined.
8. **DoLA and OPERA** are not re-implemented. Cite the paper's numbers or use the official repos.

## Research notes (for the follow-up paper)

- **VGS versus VCD.** Let Δ = log P_o − log P_d. Before its cut-off, VCD gives P ∝ P_o · exp(αΔ). VGS gives P ∝ P_o · max(1 + α·tanh(Δ/2), δ). For small Δ, log(1+α·tanh(Δ/2)) ≈ αΔ/2, so VGS behaves like VCD at half strength. For large |Δ| it saturates. The reported gains may therefore come mostly from boundedness, not from the "grounding score" framing. A bounded or clipped VCD is the obvious control.
- **Where flips can happen.** At α=1 the multiplier lies in [δ, 2]. If the greedy token has VGS ≥ 0, a rival can only win when its P_orig is more than half the greedy token's. VGS therefore mostly acts on near-ties. `score.py` reports the flip rate; relate it to confidence to explain why confident CheXagent gains about 0 and MedGemma gains the most.
- **Open questions to exploit.** A single fixed noise level for all modalities (CT, MRI, X-ray); the 2× cost; no use of attention or intermediate layers; hallucination measured only through accuracy (VASE's GREEN-based labels would be a stronger evaluation); no study of calibration or selective answering.
