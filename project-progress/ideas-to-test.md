# 02 — Ideas to Test

Ordered roughly by value per unit of effort. Each entry states the hypothesis,
the experiment, what each outcome would mean, and the cost.

Status key: **[next]** do now · **[queued]** soon · **[blocked]** waiting on
something external · **[later]** needs the above first

---

## 0. Implementation audit — run this first

`scripts/audit_implementation.py` runs four checks that isolate the possible
causes of our null result. Each one has a verdict line.

1. **Plumbing** — does the distorted image reach the model at all? Compares
   first-step logits between branches, with a black image as a control. If a
   black image changes nothing, the paired rows are mis-wired.
2. **Effective corruption** — how much noise survives the vision tower's own
   preprocessing? Measured on the actual `pixel_values` tensors.
3. **Signal strength** — how large is |VGS| on contender tokens, across a noise
   ladder from none to very strong?
4. **Headroom** — noise-independently, at what fraction of steps is a flip even
   *possible* at this alpha? This is the method's ceiling.

**A concrete defect already found by (2).** We inject noise at 512x512
(`--resize 512`, the VASE convention) but CLIP-L/14-336 resizes to 336x336
before the model sees anything. Downsampling is a low-pass filter: measured on
synthetic radiograph-like images, per-pixel noise is attenuated **~1.9x in
standard deviation** by that resize. So the corruption the model actually
receives is about half what we specified — which halves |Delta|, halves |VGS|,
and shrinks every reweighting factor.

**Fix and re-test:** `--resize 336` (or `--resize 0` to let the processor handle
it) so injected corruption equals perceived corruption. Then re-run alpha=1 and
compare flip rates. This is the single most likely implementation-side
explanation for the discrepancy, and it is cheap to check.

Note this cuts both ways as a finding: if the authors also added noise at 512
before a 336 vision tower, their effective corruption was equally attenuated and
the defect is in the original too.

---

## A. Closing the reproduction

### A1. MedGemma at full precision **[blocked — gated licence]**

**Why it matters most.** MedGemma is where the paper claims its largest gains
(+8.16 VQA-RAD, +9.12 MIMIC). It is 4B, ~11 GB at bf16, so it fits the A4000
with ~4.5 GB spare — **no quantization**, which removes the single biggest
confound in our LLaVA-Med result.

**Blocker.** HTTP 403. Accept the Health AI Developer Foundations terms at
`huggingface.co/google/medgemma-4b-it` with the same account the token belongs
to. Check `huggingface.co/settings/gated-repos` to confirm the grant.

**Experiment.** `bash scripts/run_all_vqarad.sh medgemma <mode> "<prompt>"`
after calibrating the prompt separately for this model.

**Outcomes.** Null result at full precision on their best model → the negative
finding is real and quantization is ruled out. Positive result → our LLaVA-Med
null is model-specific or a quantization artifact, and the story changes
entirely.

**Cost.** ~20 min calibration + ~3 h.

---

### A2. Noise strength sweep **[next]**

**Hypothesis.** Our flip rate (0.6% at α=1) may be low because our corruption is
weaker than the authors' — and section 0 shows at least half of it is being
filtered out by the resize before the model ever sees it. Since Eq. 2 is not literally implementable, our recipe
is an inference from VASE, not a specification. Stronger corruption → larger |Δ|
→ larger |VGS| → more flips.

**Experiment.** Sweep σ ∈ {0.03, 0.07, 0.15, 0.3} × λ ∈ {20, 70, 200}, recording
flip rate **and** accuracy at α=1. 150-question slice is enough.

**Outcomes.** If a stronger setting gives both a much higher flip rate and a
genuine gain, the discrepancy is located precisely in the noise model — a clean,
reportable finding. If flip rate rises but accuracy does not, the mechanism is
disconfirmed more strongly than by our current single setting.

**Why it's `[next]`.** Cheap, needs no external unblock, and directly tests the
most plausible explanation for the discrepancy.

**Cost.** ~4 h.

---

### A3. Trace analysis — does the core signal even exist? **[next]**

**The data already exists**: `runs/vqarad/llava-med/vgs_trace50.jsonl` holds
per-step top-k tokens with their `P_orig`, `P_dist`, `VGS` and `P_final`.
Nobody has looked at it.

**Three measurements:**
1. Do hallucinated tokens actually keep/gain probability under corruption? This
   is the paper's founding claim and it has never been independently checked.
2. What is the VGS distribution on tokens that are *contenders* (say, top-5),
   as opposed to the vocabulary at large? Note that a token moving from 1e-8 to
   1e-9 scores VGS ≈ 0.82 while being irrelevant — most large scores sit on
   tokens that cannot be selected.
3. **On questions answered wrongly, was the correct token present with positive
   VGS but insufficient probability to be promoted?**

**Why (3) is the decisive one.** If the signal is present but the reweighting is
too weak to act on it: the metric is sound, the intervention is misdesigned, and
a stronger/differently-shaped intervention is the obvious contribution (→ B2).
If the signal is absent, the premise itself fails. Either outcome is
publishable; accuracy tables alone are not.

**Cost.** ~3 h, no GPU.

---

### A4. 16-bit LLaVA-Med slice **[queued]**

**Threat being addressed.** All our LLaVA-Med runs are 4-bit because the 7B
model needs ~16.2 GB against 15.6 available. Quantization perturbs the
distributions VGS reads.

**Arguments it is not the explanation:** quantization hits both branches and VGS
depends on their ratio, so shared perturbation largely cancels; our greedy
baseline reproduces the paper's full-precision figure to within 0.4 points; the
flip rate matches the algebraic prediction, which makes no reference to
precision.

**Argument it might be:** near-ties are exactly where quantization noise matters
and exactly where VGS operates.

**Experiment.** α=1 vs `orig_pair`, 100 questions, bf16. Needs a headless
session to recover the ~1.25 GB the desktop holds, or a larger card. Attempted,
OOM'd.

**Cost.** ~2 h once memory is available.

---

### A5. CheXagent **[queued]** · A6. MIMIC-Diff-VQA **[blocked]** · A7. OPERA **[later]**

CheXagent needs its own environment (remote code, older transformers). ~5 h, and
gives a third model at full precision.

MIMIC-Diff-VQA needs PhysioNet credentialing — **start the application today**,
the wait is the binding constraint. Then subsample with `--mimic_sample_n` and
report the seed; the full 13,121 is ~10 h per configuration.

OPERA: implement or cite their numbers. ~2 days.

---

## B. Research directions (these are the actual papers)

### B1. Is boundedness the entire contribution?

**Hypothesis.** Since VGS = `1 + α·tanh(Δ/2)` and VCD = `exp(α·Δ)` agree to
first order, any advantage of VGS comes from *boundedness preventing rare
catastrophic reweightings*, not from a new notion of grounding.

**Experiment.** Implement VCD with its multiplier clipped to VGS's range
`[max(1−α, δ), 1+α]`. Compare greedy / VCD / clipped-VCD / VGS. Then sweep the
squashing function — tanh, hard clip, sigmoid — to find what actually matters.

**If clipped-VCD ≈ VGS**, the contribution is the clipping and the Visual
Grounding Score is a reparameterisation. That converts a family of methods
presented as conceptually distinct into one parameterised family with an
identified active ingredient.

**Cost.** ~3 h. Infrastructure already exists.

---

### B2. Confidence-aware intervention

**The gap it addresses.** At α=1 the multiplier is bounded in [δ, 2], so a rival
token can only overtake the greedy choice if it already holds **more than half**
the greedy token's probability. VGS is structurally confined to near-ties, and
near-ties are rare — hence 0.6%. This is a design flaw, identified by
measurement rather than speculation.

**The idea.** Make α a function of the entropy or margin of `P_orig`. Where the
model is confident, no correction is possible or needed; where it is uncertain,
apply a correction far stronger than a global α could safely allow. The flip
rate becomes the tuning instrument — it directly reports whether the method is
doing anything.

**Extension.** Connects naturally to selective prediction: a model that knows
when it is uncertain can also *decline to answer*, which is the clinically
appropriate behaviour and is addressed by no method in this literature.

**Cost.** ~1 week including evaluation.

---

### B3. An audit of medical VQA evaluation

**The observation.** Two artifacts we documented are not specific to this paper
— they are conventions inherited across the literature:

1. Substring matching credits answers asserting the *opposite* of the ground
   truth (6.3% of closed questions in our slice).
2. Recall-based open scoring rewards verbosity independently of correctness
   (every method we tested gained open recall while losing closed accuracy).

**The contribution.** Quantify how much published closed-ended accuracy is
spurious substring credit across several models; show the length effect
directly; propose verdict-extracting and length-controlled alternatives;
re-evaluate several published methods under the corrected protocol and report
how rankings change.

**Why this one is most likely to finish in a MIDL/MICCAI cycle.** No new method,
no new data, only careful measurement — and the tooling already exists in this
repo.

**Cost.** ~2 weeks.

---

### B4. Smaller ideas worth a paragraph each

- **Length-controlled open metric.** Replace recall with F1, or match generation
  length across methods, and re-run everything. Would show how much of the
  paper's "+8.98 open-ended recall" survives. (~3 h)
- **Modality-specific noise.** One fixed σ, λ for CT, MRI and X-ray is
  unjustified — these have entirely different noise characteristics. Does
  per-modality calibration help? (~1 day)
- **Cheaper than 2×.** Our paired-row batching already gives ~16% overhead
  instead of 2×. Can the distorted branch be evaluated every *k* steps, or only
  at high-entropy steps, for near-free correction? (~2 days)
- **Attention-based grounding.** Instead of corrupting the image, read
  cross-attention to image tokens directly. No second forward pass at all.
  (~1 week)

---

## C. Questions for the authors

Worth emailing early and politely; they may simply answer.

1. **What fraction of decoding steps does reweighting change the selected
   token?** We measure 0.6% at α=1. If they see substantially more, the
   discrepancy is located and everything follows from it.
2. What exactly is the noise operation? Eq. 2 as written saturates the image.
   Poisson resampling at scale λ? Applied before or after resize/normalise?
3. What prompt was used?
4. How is a sentence-length answer matched against a one-word ground truth?
5. Was the greedy baseline run with one image per batch, or the same two-row
   batch shape as VGS? (Affects ~7.5% of answers for us.)
6. How were the 13,121 MIMIC samples selected, and how were two-image questions
   fed to single-image models?
7. What is "VGS metric only" in Table 2? The metric alone does not specify a
   decoding rule.
8. Which LLaVA-Med checkpoint and code path?
9. What explains DoLA moving from −6.30 (LLaVA-Med) to +7.06 (MedGemma) on the
   same dataset?

---

## D. Framing note

If the reproduction ultimately fails across models and precisions, **do not
frame the paper as an attack.** The right framing: the method is a
well-motivated idea whose mechanism, on analysis, is confined to a narrow
regime; here is the measurement showing how narrow; here is the corrected
evaluation protocol; here is a modification that addresses the limitation. That
is a contribution to the field rather than a complaint about a paper, and it is
considerably more likely to be accepted.
