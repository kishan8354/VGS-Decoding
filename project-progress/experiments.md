# 03 — Experiments and Results

All results below: LLaVA-Med (`llava-med-v1.5-mistral-7b-hf`), **4-bit**,
VQA-RAD, NVIDIA RTX A4000 (15.6 GB), prompt `"{question} Answer briefly."`,
`--max_new_tokens 48`. Every run file has a `.config.json` sidecar recording the
exact arguments.

Coverage against the paper: **1 of 6 model-dataset cells**, at a precision they
did not use.

| Model | VQA-RAD | MIMIC-Diff-VQA |
|---|---|---|
| LLaVA-Med | done (4-bit only) | not done |
| CheXagent | not done | not done |
| MedGemma | **blocked** (gated licence) | not done |

---

## E0. Unit tests — 13 passed

CPU, ~6 s, no downloads. Small randomly-initialised models.

- **Maths (4).** `tanh` form equals the literal ratio to 1e-5; [-1,+1] bound;
  underflow and zero-probability cases finite with correct limits; reweighting
  equals a direct implementation of Eq. 5; α=0 returns the original distribution
  exactly; VCD checked on a hand-computed example including its plausibility mask.
- **Decoding (4).** Paired rows stay in lock-step; batched generation matches a
  cache-free stepwise re-implementation token for token (VGS at two strengths,
  VCD); null config and α=0 reproduce plain greedy exactly; at high α the flip
  count is positive, so an inert implementation fails.
- **Architecture (1).** Same invariants on a tiny Gemma-3 (MedGemma's class).
  *Note: initially vacuous — Gemma's projector is zero-initialised, so the image
  had no influence. Fixed by re-initialising it.*
- **DoLA (2).** Hook install/removal, layer selection in the candidate set,
  output differs from greedy, layers found in the LM not the vision tower.
- **Metrics (2).** Each matching rule on adversarial cases ("Normal study" must
  not match "no" under `word`); overall metric verified against the paper's own
  arithmetic.

---

## E1. Baseline calibration — **the implementation is correct**

The first 50-question attempt gave closed accuracy 43.75 against the paper's
68.92 — a 25-point gap. Hypothesis: prompt and matching rule, not a bug.

4 prompts × 3 matching rules, n=150, ranked by distance to the paper's greedy
figures (open 34.45 / closed 68.92 / overall 53.64):

| Prompt | Match | Open | Closed | Overall | mean abs Δ |
|---|---|---|---|---|---|
| **brief** | **substring** | **37.98** | **68.35** | **53.98** | **2.05** |
| bare | substring | 39.31 | 69.62 | 55.27 | 2.78 |
| vase | substring | 38.62 | 64.56 | 52.28 | 4.27 |
| brief | word | 37.98 | 62.03 | 50.65 | 5.21 |
| brief | first | 37.98 | 60.76 | 49.98 | 5.85 |
| bare | word | 39.31 | 62.03 | 51.27 | 5.88 |
| short | substring | 40.45 | 62.03 | 51.81 | 6.45 |
| bare | first | 39.31 | 60.76 | 50.60 | 6.51 |
| vase | word | 38.62 | 58.23 | 48.95 | 7.43 |
| vase | first | 38.62 | 58.23 | 48.95 | 7.43 |
| short | word | 40.45 | 59.49 | 50.48 | 7.71 |
| short | first | 40.45 | 59.49 | 50.48 | 7.71 |

Prompts: `vase` = VASE's "Answer this question as concisely as possible based on
the provide images: {q}"; `short` = "{q}\nAnswer the question using a single word
or phrase."; `brief` = "{q} Answer briefly."; `bare` = the question alone.

**Findings.**
- Closed within 0.6 and overall within 0.34 of the paper. The 25-point gap was
  entirely prompt + matching. **The implementation was never wrong.**
- Matching rule moves closed accuracy by >6 points; prompt by ~5. Both larger
  than the 4.11-point effect the paper attributes to its method.
- The rule that best reproduces the paper is the lenient one → investigated in
  E2.

---

## E2. The substring matching flaw

Every closed question credited by `substring` but not `word`, n=150 (5 of 79
closed questions):

| Truth | Model's answer |
|---|---|
| no | "…the anatomy of the brain gyri appears to be **no**rmal." |
| no | "**Yes**, the chest X-ray is considered safe… a **no**n-invasive imaging t…" |
| no | "**Yes**, the mass in the image is likely representing a **ne**oplastic process…" |
| no | "It appears that the image is related to a **ne**oplastic process. however…" |
| no | "**Yes**, the chest X-ray shows an ab**no**rmally large heart… cardiomegaly" |

The rule matches `n-o` inside `normal`, `non-invasive`, `neoplastic`,
`abnormally`. **Three of five begin with "Yes"** and are credited as "no".

5/79 = 6.3 points, which is exactly the gap between 68.35 (substring) and 62.03
(word). Since substring is what reproduces the paper's 68.92, their figure
likely contains comparable spurious credit.

**Policy adopted:** report every result under both a lenient and a strict rule.

---

## E3. Main results — full VQA-RAD, n=451

8 configurations, ~2.5 h total.

### Against plain greedy, substring matching

| Method | Open | Closed | Overall | Δ | Flip% | p_boot | p_McN |
|---|---|---|---|---|---|---|---|
| Greedy | 34.99 | 67.73 | 53.21 | — | — | — | — |
| VCD | 36.78 | 65.74 | 52.90 | −0.32 | 1.3 | 0.657 | 0.180 |
| DoLA | 36.12 | 67.33 | 53.49 | +0.28 | — | 0.404 | 1.000 |
| VGS α=0.5 | 35.43 | 66.93 | 52.96 | −0.25 | 0.4 | 0.694 | 0.625 |
| VGS α=1.0 | 36.35 | 66.93 | 53.37 | +0.16 | 0.6 | 0.383 | 0.625 |
| VGS α=1.5 | 36.73 | 66.93 | 53.54 | +0.33 | 0.8 | 0.275 | 0.625 |
| VGS α=2.0 | 36.58 | 66.93 | 53.47 | +0.26 | 1.2 | 0.359 | 0.727 |

Greedy reproduces the paper (53.21 vs 53.64, difference 0.43 where one question
= 0.22). **Nothing else does:** VGS +0.16 vs their +4.11; VCD −0.32 vs their
−5.93; DoLA +0.28 vs their −6.30. No p-value near significance.

### The baseline above is wrong — use `orig_pair`

`orig_pair` runs the full paired machinery but discards the distorted branch, so
it is mathematically identical to greedy. It agrees with greedy on only
**92.46%** of answers — **34 of 451 differ**.

Diagnosis (`scripts/diagnose_pairing.py`): **zero verdict changes, zero empty
outputs**, divergence 13–38 words into a sentence. Example:

> **Q:** what are these opacities anterior to the right kidney?
> **Greedy:** "…caused by bleeding *or injury.*"
> **Paired:** "…caused by bleeding *due to an…*"
> *(diverges after 36 identical words; verdict unchanged)*

This is floating-point non-determinism from the changed batch shape, not a bug.
But it affects **more answers than VGS itself changes**, so comparing a paired
method against single-row greedy conflates method with artifact.

### Against `orig_pair`, substring matching ← **the correct comparison**

| Method | Open | Closed | Overall | Δ | Flip% | p_boot | p_McN |
|---|---|---|---|---|---|---|---|
| orig_pair | 35.18 | 68.53 | 53.74 | — | 0.0 | — | — |
| VGS α=1.0 | 36.35 | 66.93 | 53.37 | **−0.37** | 0.6 | 0.756 | 0.125 |
| VGS α=1.5 | 36.73 | 66.93 | 53.54 | **−0.20** | 0.8 | 0.637 | 0.125 |
| VGS α=2.0 | 36.58 | 66.93 | 53.47 | **−0.27** | 1.2 | 0.642 | 0.289 |

**The sign flips.** α=1.5 looked like +0.33 against greedy; it is −0.20 against
the matched baseline. The apparent gain was batch-shape noise.

*Implication for the paper:* if they baselined against single-row greedy — which
the natural implementation would — part of their effect is the same artifact.
Their +4.11 is far larger than this ±0.5, so it is not explained away, but their
comparison was not clean either.

### Against `orig_pair`, strict (`first`) matching

| Method | Open | Closed | Overall | Δ | p_boot | p_McN |
|---|---|---|---|---|---|---|
| orig_pair | 35.18 | 58.17 | 47.98 | — | — | — |
| VCD | 36.78 | 55.78 | 47.35 | −0.62 | 0.798 | 0.070 |
| DoLA | 36.12 | 56.57 | 47.50 | −0.47 | 0.670 | 0.289 |
| VGS α=1.0 | 36.35 | 56.97 | 47.83 | −0.15 | 0.618 | 0.250 |

All negative under the strict rule too. Note closed accuracy drops ~10 points
across the board (68.53 → 58.17 for the baseline) once the substring artifact is
removed — that is how much of the published figure is spurious credit.

---

## E4. The verbosity artifact

Change relative to `orig_pair`, substring matching:

| Method | Δ Open | Δ Closed |
|---|---|---|
| VCD | **+1.60** | −2.79 |
| DoLA | **+0.94** | −1.20 |
| VGS α=1.0 | **+1.17** | −1.60 |
| VGS α=1.5 | **+1.55** | −1.60 |
| VGS α=2.0 | **+1.40** | −1.60 |

**Every** method gains open recall and loses closed accuracy. Too consistent to
be chance. Open scoring is pure recall, so longer answers can only score higher;
perturbing the distribution makes answers longer and more hedged, and is rewarded
for it without any gain in correctness.

**Therefore: a gain in open recall is not evidence of reduced hallucination.**
The paper's abstract highlights "+8.98 in open-ended recall" — this artifact
bears directly on how that should be read.

---

## E5. Mechanism — why nothing happened

The processor logs, per step, whether reweighting changed the selected token.

| Method | Flip rate | Overall Δ |
|---|---|---|
| orig_pair | 0.0% | — |
| VGS α=0.5 | 0.4% | −0.25 |
| VGS α=1.0 | **0.6%** | −0.37 |
| VGS α=1.5 | 0.8% | −0.20 |
| VGS α=2.0 | 1.2% | −0.27 |
| VCD α=1.0 | 1.3% | −0.32 |

At default strength VGS changes the model's choice in **6 steps per 1000**. The
rate rises with α exactly as predicted, confirming the implementation is active.
`orig_pair` at 0.0% is a correctness check.

**This is forced by the algebra.** Let `g` be the greedy token, `r` a rival.
Since VGS ∈ [−1,+1], the multiplier lies in `[max(1−α, δ), 1+α]`. For the common
case where the greedy token is neutral (VGS≈0) and the rival is maximally
grounded (VGS≈+1), at α=1 the rival wins only if

```
P_orig(r) > P_orig(g) / 2
```

**The rival must already hold more than half the greedy token's probability.**
Models generating short factual answers are usually far more confident than that.
VGS is *structurally* confined to near-ties, and near-ties are rare. 0.6% is not
a weak implementation — it is what the algebra predicts.

**Arithmetic check on a 4-point claim.** 4 points of 451 questions ≈ 18 answers
flipping wrong→right. At 0.6% over ~30-token answers, VGS changes ~0.18 tokens
per answer ≈ 81 token changes across the whole set. For those to net +18 correct
answers, a large fraction of all flips would have to land on the decisive token
*and* be correct. Possible in principle; implausible at the claimed scale.

Two readings remain, distinguished by experiment not argument: either their
configuration produces a much higher flip rate (→ noise sweep, `02-ideas.md` A2),
or the improvement comes from something other than reweighting (→ verbosity
artifact, E4).

---

## E6. Timing — the 2× overhead claim is pessimistic

Full 451-question runs, LLaVA-Med 4-bit:

| Method | Total | Per question |
|---|---|---|
| Greedy | 18:19 | 2.44 s |
| orig_pair | 20:32 | 2.73 s |
| VCD | 20:42 | 2.75 s |
| DoLA | 23:15 | 3.09 s |
| VGS α=1.0 | 21:27 | 2.85 s |

Paired methods cost **~16% more, not 100%**. Our paired-row design puts both
forward passes in one batch, so the GPU absorbs most of the second. A modest but
genuine improvement on the paper's own cost analysis.

DoLA is slowest despite needing no second image pass — the per-step JSD over
candidate layers is itself expensive.

---

## Summary of findings

| # | Finding |
|---|---|
| F1 | Greedy baseline reproduces the paper to within 0.43 overall points |
| F2 | VGS shows no benefit at any α: −0.15 to −0.37 vs a matched baseline, two matching rules, none significant |
| F3 | VCD and DoLA also fail to reproduce their reported large degradations |
| F4 | The claimed monotone α response is absent — observed curve is flat within 0.4 points |
| F5 | Batch shape alone changes 34/451 answers, exceeding the method's own effect |
| F6 | Substring matching credits answers asserting the opposite of the truth, 6.3% of closed questions |
| F7 | Every method gains open recall and loses closed accuracy — a length artifact |
| F8 | Flip rate of 0.6% at α=1, consistent with the algebraic bound |
| F9 | Paired decoding costs ~16% extra, not 2× |

F5, F6 and F8 are robust regardless of whether the method works — they are
properties of paired decoding, of the metric, and of the algebra respectively.

---

## E7. Implementation audit — why we may not be reproducing

`scripts/audit_implementation.py` isolates four possible causes. **Not yet run
against real weights** — this is the immediate next action.

**One defect already found by code inspection.** Noise is injected at 512x512
(VASE convention, our `--resize 512` default) but CLIP-L/14-336 downsamples to
336x336 before the model sees anything. Downsampling low-pass filters
independent per-pixel noise; measured on synthetic radiograph-like images the
attenuation is **~1.9x in standard deviation**:

| stage | std of (dist − orig) |
|---|---|
| injected at 512 | 0.1053 |
| surviving after resize to 336 | 0.0563 |
| (noise injected directly at 336) | 0.1064 |

So the model receives roughly half the corruption we specified. Halved |Δ| →
halved |VGS| → every reweighting factor shrinks toward 1 → fewer flips. This is
the most likely implementation-side explanation for the null result and it is
cheap to test: re-run with `--resize 336` and compare flip rates.

It cuts both ways as a finding. If the authors also injected noise at 512 before
a 336 vision tower, their effective corruption was attenuated identically and
the defect is in the original method as published.

**To run:**

```bash
python scripts/audit_implementation.py --model llava-med --load_in_4bit --limit 20
```

Then, if CHECK 2 or 3 flags weak corruption:

```bash
bash scripts/run_all_vqarad.sh llava-med substring "{question} Answer briefly." \
  --load_in_4bit --resize 336
```

## Not yet done

| Item | Blocker | Cost once unblocked |
|---|---|---|
| MedGemma, full precision | gated licence | ~3 h |
| Noise strength sweep | none — **do next** | ~4 h |
| Trace analysis (data captured) | none — **do next** | ~3 h |
| 16-bit LLaVA-Med slice | GPU memory (needs headless) | ~2 h |
| CheXagent | separate environment | ~5 h |
| MIMIC-Diff-VQA | PhysioNet credentialing (weeks) | ~1 day + subsampling |
| OPERA baseline | not implemented | ~2 days |
| Figures | matplotlib now installed; rerun `make_figures.py` | ~10 min |
| **Implementation audit** | none — **do first** | ~30 min |
| **Re-run with `--resize 336`** | none — **do first** | ~2.5 h |

**The negative result is not yet publishable as one.** One model, one dataset,
one precision. If the effect is also absent for MedGemma at full precision, a
failed reproduction plus a mechanistic explanation is a stronger contribution
than confirming their table would have been.

---

## Run inventory

| File | Contents |
|---|---|
| `runs/calib/llava-med_{vase,short,brief,bare}.jsonl` | 150 q each, four prompts (E1) |
| `runs/vqarad/llava-med/greedy.jsonl` | 451 q, baseline |
| `runs/vqarad/llava-med/orig_pair.jsonl` | 451 q, null config |
| `runs/vqarad/llava-med/vcd.jsonl` | 451 q, VCD α=1 |
| `runs/vqarad/llava-med/dola.jsonl` | 451 q, DoLA high layers |
| `runs/vqarad/llava-med/vgs_a{0.5,1.0,1.5,2.0}.jsonl` | 451 q each, α sweep |
| `runs/vqarad/llava-med/vgs_trace50.jsonl` | 50 q, per-token VGS values — **unanalysed** |

Record format:

```json
{"qid": "vqarad-test-00000",
 "question": "is there evidence of an aortic aneurysm?",
 "answer": "yes", "is_closed": true,
 "pred": "No, there is no evidence of an aortic aneurysm ...",
 "sec": 2.44, "steps": 31, "flips": 0, "meta": {}}
```

`steps` and `flips` are what make E5 possible — the most informative addition
this implementation makes over what the paper describes.
