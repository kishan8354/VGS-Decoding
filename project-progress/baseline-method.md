# 01 — The Baseline Method and How It Is Evaluated

Reference: Kolli et al., *VGS-Decoding: Visual Grounding Score Guided Decoding for
Hallucination Mitigation in Medical VLMs*, arXiv:2603.20314v1, 19 Mar 2026.

This note covers what the method does, what it is compared against, and — at
length, because it turned out to matter more than anything else — how the
evaluation is actually computed.

---

## 1. The method

### The observation it rests on

When the input image is corrupted, a token that was using visual evidence loses
probability. A token produced from language priors keeps its probability or
gains, since mass shed by grounded tokens is redistributed. That asymmetry is a
signature separating the two, computable from quantities the model already
produces — no labels, no retraining, no external verifier.

### The score

With `P_o(t)` the probability of token `t` given the original image and `P_d(t)`
its probability given a corrupted version, at the same step with the same prefix:

```
VGS(t) = (P_o(t) - P_d(t)) / (P_o(t) + P_d(t))
```

Bounded in [-1, +1]. Positive means grounded (lost probability under corruption),
negative means suspected hallucination, near zero means indifferent to the image
— typical of function words.

**Implementation note.** Computed as `tanh((log P_o - log P_d) / 2)`, which is
algebraically identical but does not underflow across a large vocabulary. See
`vgsd/vgs.py`.

### The intervention

```
P_final(t) ∝ P_o(t) × max(1 + α·VGS(t), δ)          δ = 0.01 default
```

Renormalise, take the argmax. Grounded tokens are amplified, suspected
hallucinations suppressed, neutral tokens left alone.

### The noise

The paper writes `V' = V + N(0, σ²) + Poisson(λ)` with σ=0.07, λ=70. **This is
not literally implementable** — adding a Poisson(70) sample to a [0,1] image
saturates it. We follow VASE (MICCAI 2025, the paper's ref [19], which uses
exactly these values and is almost certainly the codebase this was built on):

```
x ← clamp(x + N(0, 0.07²), 0, 1)
x ← clamp(Poisson(clamp(70x, 0, 255)) / 70, 0, 1)
```

per RGB channel after a 512×512 resize. Seeded per question, so runs are
reproducible and compared methods see identical corruption.

### Cost

Two forward passes per step. We batch them as two rows of one pass, so measured
overhead is ~16%, not the 2× the paper states. See `03-experiments.md`.

---

## 2. What it is compared against

| Method | Second distribution from | Combination | Bounded? | Status here |
|---|---|---|---|---|
| Greedy | — | — | — | Implemented |
| VCD | noise-corrupted image | `(1+α)·log P_o − α·log P_d`, plausibility-masked | No | Implemented |
| DoLA | an earlier transformer layer | same subtractive form, JSD layer choice | No | Re-implemented |
| OPERA | beam search + over-trust penalty | — | — | **Not implemented** |
| VGS | noise-corrupted image | `P_o × (1 + α·tanh(Δ/2))` | Yes | Implemented |

**DoLA note.** transformers ≥4.56 moved its built-in DoLA to a remote
`custom_generate` repo needing `trust_remote_code` and network access at
generation time. Unsuitable for a reproducibility artifact, so it is
re-implemented locally in `vgsd/dola.py` using forward hooks, per-step JSD layer
selection, the plausibility mask and repetition penalty 1.2.

**OPERA note.** Needs beam search with the over-trust penalty and
retrospection-allocation. Substantial work for a baseline the paper itself argues
is unsuited to short answers. Use their repo or cite their numbers.

### VGS is a bounded VCD

Writing `Δ = log P_o − log P_d`, both methods multiply `P_o` by a function of Δ:

- VCD: `exp(α·Δ)` — unbounded
- VGS: `max(1 + α·tanh(Δ/2), δ)` — saturates at `1+α`

For small Δ, `log(1 + α·tanh(Δ/2)) ≈ αΔ/2` — VCD at half strength. They differ
only in the tails. So VGS is best described as a *bounded, half-strength VCD*
rather than a different kind of signal. The obvious control the paper never runs
— clipped VCD — is in `02-ideas.md`.

---

## 3. Evaluation — read this section before trusting any number

Three metrics, following LLaVA-Med convention.

### Overall (settled)

The sample-weighted mean of the two below. Verified against the paper's own
arithmetic: `(200 × 34.45 + 251 × 68.92) / 451 = 53.64`, exactly their reported
figure. No ambiguity here.

### Open-ended recall (has an artifact)

```
recall = |tokens(gt) ∩ tokens(pred)| / |tokens(gt)|
```

multiset intersection after lowercasing and stripping punctuation.

**Recall, not precision or F1.** A longer answer can only score equal or higher,
since extra words can only add matches. Consequence: *any* intervention that
makes the model more verbose gains open recall without improving correctness.
Observed empirically — every method we tested gained open recall and lost closed
accuracy. A gain in open recall is **not** by itself evidence of reduced
hallucination.

### Closed-ended accuracy (three rules, and the choice moves it 6+ points)

The models answer in sentences; ground truth is one word. Matching rule:

| Rule | Definition | Closed acc (brief prompt, n=150) |
|---|---|---|
| `substring` | gt string appears anywhere | 68.35 |
| `word` | gt appears as a whole word | 62.03 |
| `first` | first yes/no word must equal gt | 60.76 |

`substring` is LLaVA-Med's own convention and the rule that best reproduces the
paper's 68.92. **It is also demonstrably broken.** Every case it credits and
`word` does not, from a 150-question slice:

| Truth | Model's answer |
|---|---|
| no | "…the anatomy of the brain gyri appears to be **no**rmal." |
| no | "**Yes**, the chest X-ray is considered safe… It is a **no**n-invasive…" |
| no | "**Yes**, the mass is likely representing a **ne**oplastic process…" |
| no | "It appears the image is related to a **ne**oplastic process…" |
| no | "**Yes**, the chest X-ray shows an ab**no**rmally large heart…" |

The rule matches the letters `n-o` inside `normal`, `non-invasive`, `neoplastic`,
`abnormally`. Three of the five answers *begin with "Yes"* and are credited as
"no". Five of 79 closed questions = 6.3 points, which is the entire gap between
the lenient and strict scores.

Since `substring` is what reproduces the paper's number, the most economical
explanation is that the original authors used it too — meaning their reported
68.92 contains comparable spurious credit. Their internal comparison is still
valid (all methods inflated equally), but the absolute figures overstate clinical
correctness.

**Our policy: report both rules for every experiment.** Lenient for
comparability with published numbers, strict as the corrected figure. A
conclusion that holds under both is robust to this artifact.

### Significance

Paired bootstrap (10k resamples) on the continuous overall score; exact McNemar
on closed-ended correctness. Both paired, since methods see identical questions.
With 451 questions, one question is worth 0.22 points — differences under a point
need a p-value attached.

---

## 4. The baseline you compare against also matters

Paired methods put two rows in the batch, which changes matrix shapes, which
changes floating-point accumulation order. Under **identical mathematics**, the
null configuration (`orig_pair` — full paired machinery, distorted branch
discarded) differs from single-row greedy on **34 of 451 answers**.

That is larger than the effect VGS itself produces. Diagnosis confirms it is
numerics, not a bug: zero verdict changes, zero empty outputs, divergence 13–38
words into a sentence.

**So: always baseline against `orig_pair`, never plain greedy.** Against greedy,
VGS α=1.5 looks like +0.33; against `orig_pair` it is −0.20. The apparent gain
was batch-shape noise.

---

## 5. What the paper does not specify

Each was an unavoidable decision point with no guidance in the text:

1. The noise operation (Eq. 2 is not literally implementable)
2. Whether noise is applied before or after resize/normalise
3. **The prompt** — moves closed accuracy by ~5 points
4. **The matching rule** — moves closed accuracy by ~6 points
5. How the 13,121 MIMIC-Diff-VQA samples were selected
6. How two-image difference questions are fed to single-image models
7. Which LLaVA-Med checkpoint (official repo vs HF conversion)
8. Baseline hyperparameters (VCD's β, DoLA's layer preset)

Items 3 and 4 each move the number by more than the 4.11-point effect the paper
attributes to its method. Questions for the authors are listed in
`02-ideas.md`.

---

## 6. Datasets

**VQA-RAD** — 315 radiology images, 451 test questions (200 open / 251 closed).
Split verified to match the paper exactly. Free, small, fast: one configuration
runs in ~20 minutes.

**MIMIC-Diff-VQA** — differences between sequential chest X-rays, 13,121 test
samples (8,380 open / 4,741 closed). Requires PhysioNet credentialing (CITI
course, data use agreement — weeks). 29× larger than VQA-RAD, so one
configuration is ~10 hours; subsample with `--mimic_sample_n` and report the
seed.
