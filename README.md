# DeepShield

**Personal Deepfake Identity Protection & Multi-Signal Detection System**

DeepShield applies independent protection layers to a user's face images, then
analyses suspect images and videos for content biometrically similar to that
user, fusing several forensic signals into one explainable risk assessment.

**Status: Phases 0–15 implemented and measured.** Face detection, alignment,
embedding, enrollment, matching, image and video analysis, deepfake adapters,
fingerprinting, watermarking, the risk engine, the protection pipeline,
adversarial cloaking research, the robustness benchmarks, the REST API and
provenance all run. Every number in this README was produced by the code in this
repository; the commands that produce them are shown.

**Read [What Does Not Work](#what-does-not-work) before using or extending
this.** Three of the five evidence signals are working and measured; one is
computed but deliberately excluded from scoring, and one protection layer was
measured to be ineffective.

---

## What Does Not Work

Everything below is a measured result, not an unfinished feature. The code paths
exist, are tested, and refuse to overstate themselves — that is why they are
listed here rather than hidden behind an optimistic default.

### Signal status

| Signal | Status | Measured | Notes |
|---|---|---|---|
| Identity similarity | ✅ works | EER 0.0000, 850 genuine vs 24 650 impostor | scored, weight renormalised |
| Watermark attribution | ✅ works | 0 wrong codes in 2 550 measurements | scored |
| Perceptual fingerprint | ✅ works | exact + near-duplicate matching | scored above evidence threshold only |
| Provenance | ✅ works | local, self-asserted lineage | scored |
| **Synthetic-media (deepfake)** | ⛔ **not scored** | best of 4 attempts: **AUC 0.59** | computed and reported, excluded from the risk score |
| **Adversarial cloaking** | ⛔ **ineffective** | displaces embeddings **less than JPEG does** | disabled by default, kept as research |

### 1. The deepfake signal is excluded from the risk score

Four attempts, none usable:

| Attempt | ROC-AUC | False positives on real photos |
|---|---|---|
| Hand-built blending detector | 0.579 (0.641 in-sample) | — |
| `Wvolf/ViT_Deepfake_Detection` | 0.594 | **76.3%** |
| `dima806/deepfake_vs_real_image_detection` | 0.586 | **70.4%** |
| `prithivMLmods/Deep-Fake-Detector-Model` | 0.385 (below chance) | 45.6% |

The two best published models call **seven out of ten genuine photographs
synthetic**. The risk engine therefore refuses to score the signal, and
`evaluate_deepfake_detectors.py --write` refuses to adopt any of them.

**What unblocks it:** a detector trained on a real labelled corpus
(FaceForensics++, Celeb-DF, DFDC). All three require a signed data-use
agreement, which is the one step this repository cannot take by itself. Once a
model exists, everything else is already built:

```bash
python scripts/fetch_deepfake_detector.py --model <hf-repo-or-path>
python scripts/evaluate_deepfake_detectors.py --write   # adopts it only if it clears the bar
```

The bar is ROC-AUC ≥ 0.75 with a false positive rate ≤ 10%.

### 2. Adversarial cloaking does not protect

At a perceptually acceptable budget (ε=0.03) the cloak moves a face embedding by
0.016 cosine distance, while plain JPEG-70 moves it 0.022 and a 50% resize moves
it 0.046. Cross-model transfer is three times weaker again. At ε=0.06 the image
is visibly degraded (SSIM 0.873) and recognition still succeeds at 0.97
similarity.

It is `enabled: false` by default and documented as research. **This is a
finding, not a bug** — do not enable it expecting protection.

### 3. Watermark limits

Recovered: JPEG q90/q70, WebP, blur, noise, brightness, contrast, screenshot,
resize 75%, crop 10% and 20%. **Not recovered: rotation, resize 25%, and — on
small images — crop 30% and JPEG q50.** Resolution matters: crop 30% survives on
a 512-pixel image but not on a 250-pixel one.

**What unblocks rotation:** log-polar or Fourier-Mellin synchronisation, or a
rotation sweep added to the existing grid search. The latter is ~30 minutes of
work but multiplies the search volume, and the search volume is exactly what
produced a false attribution during development — so it needs the same
measurement discipline, not just the code.

### 4. Calibration is provisional

Thresholds were fitted on 30 identities from Labeled Faces in the Wild, a
benchmark skewed by demographic and pose, with non-independent within-identity
comparisons.

**What unblocks it:** refit on photographs resembling the deployment population.
This takes minutes once such photos exist:

```bash
python scripts/build_evaluation_set.py       # or point it at your own labelled set
python scripts/calibrate_thresholds.py --write
```

### 5. Out of scope by design

- **Training-data attribution** — cannot be proven by any technique here, and
  the system never claims it.
- **Web monitoring / crawling** (Phase 16) — interfaces only; input is what the
  user submits.
- **Full C2PA** — a declared stub. `verify()` returns `supported: false` so a
  caller can tell "no credential" from "cannot check".

### Before deploying for a real user

- [ ] Refit thresholds on representative photographs (§4 above)
- [ ] Obtain and wire a deepfake detector that clears the bar (§1 above), or
      accept that the synthetic-media signal is absent
- [ ] Leave adversarial cloaking disabled unless you have re-measured it
- [ ] Decide whether the watermark's rotation gap matters for your channels

---

## Project Goal

Answer one research question with real code and real measurements:

> Can we apply protection to a user's face images, then detect content with high
> facial similarity to that user, and assess how likely that content is
> synthetic — using several independent pieces of evidence rather than one?

The goal is **not** a program that "stops deepfakes". It is a personal identity
protection system at the intersection of identity protection, multimedia
forensics, adversarial ML and content provenance.

---

## Threat Model

**Who we protect.** An individual who publishes face photos publicly.

**What the adversary does.** Collects those photos, produces synthetic images or
videos of the person, redistributes them. They can re-encode, crop, resize,
screenshot, colour-correct or AI-upscale anything, and can deliberately try to
strip watermarks and defeat cloaking.

**What we control.** The images the user protects before publishing, and the
content later submitted for analysis.

**What we do not control.** The adversary's model and pipeline, and any copy of
the user's photos published before enrollment.

**Consequence.** Every single signal can be destroyed independently — this is
measured below, not assumed. That is why the system is built as multi-signal
evidence fusion rather than as one detector.

---

## What This System Can Detect

- Whether a face in submitted media has **high biometric similarity** to an
  enrolled identity.
- Whether a **DeepShield watermark** is recoverable, and which distribution
  channel it was issued to.
- Whether a file is an **exact or perceptually near-duplicate** of a registered
  protected asset.
- A **synthetic-media likelihood score** from a pluggable detector.
- A combined, explainable **risk score** stating which evidence moved it.

Correct phrasing for a positive result, and what the code actually prints:

> Content containing a face with high similarity (0.995) to the enrolled identity
> 'curie' was found. The synthetic-media detector scored it 0.763; this is a
> likelihood, not a determination. A DeepShield watermark (46940c50) was
> recovered, indicating the file descends from a registered protected asset.

---

## What This System Cannot Prove

> **This system cannot prove that a particular image was used as training data
> for a generative model.**

It also cannot establish the following, and the code never claims them:

| Claim | Why it does not hold |
|---|---|
| "A watermark survives into a deepfake generated from the image." | A watermark protects the file, not the identity. Generative models do not reproduce it. |
| "Adversarial perturbation works against every model." | Measured below: cloaking at acceptable quality moves embeddings *less than JPEG compression does*. |
| "The deepfake detector catches every generator." | Detectors generalise poorly to unseen generator families. Ours is uncalibrated and excluded from the score. |
| "High face similarity means the content is a deepfake." | Similarity is an identity signal. Real photos of the user score highest of all. |
| "A high deepfake score means the user's photo was training data." | Different problem entirely — that is membership inference. |
| "A perceptual hash can track an image into a generated output." | pHash tracks copies of a file, not identities inside a model. |

**Problem A (training-data attribution)** is out of scope.
**Problem B (identity similarity detection)** is the MVP, and it works.

---

## Architecture

```text
                         ┌─────────────────┐
                         │      USER       │
                         └────────┬────────┘
                     ┌────────────▼────────────┐
                     │   Identity Enrollment   │
                     │     Face Embeddings     │
                     └────────────┬────────────┘
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
    ┌─────────────────┐                      ┌─────────────────┐
    │ Image Protection│                      │ Identity Store  │
    └────────┬────────┘                      └─────────────────┘
     ┌───────┼─────────────┐
     ▼       ▼             ▼
 Watermark Fingerprint Adversarial
     └───────┼─────────────┘
             ▼
      Protected Content

===========================================================

              Suspected Image / Video
                        ▼
        Media Preprocessing · hash · fingerprint · watermark
                        ▼
            Face Detect → Align → Embed
                        ▼
                 Identity Search
                        ▼
                 Candidate Filter      ← expensive detectors run only past here
             ┌──────────┼────────────┐
             ▼          ▼            ▼
         Deepfake    Watermark    Fingerprint
         Detector    Detector      Matcher
             └──────────┼────────────┘
                        ▼
                 Provenance Check
                        ▼
              Evidence Aggregation
                        ▼
                   Risk Engine
                        ▼
                Evidence Report
```

**Candidate gating** is what makes this affordable and what keeps false
positives down: identity similarity runs first and cheaply, and only faces above
the candidate threshold reach the deepfake detector. A synthetic image of a
stranger is not this user's problem, and scoring it would only add noise.

### Layout

```text
src/deepshield/
├── cli.py            command surface and exit codes
├── config.py         typed YAML configuration; no hard-coded thresholds
├── logging_utils.py  logging plus biometric redaction
├── registry.py       name → backend factories, keeps models swappable
├── types.py          shared data structures and their JSON shapes
├── media.py          decoding, saving, hashing
├── quality.py        PSNR, SSIM, sharpness
├── transforms.py     the shared transformation engine
├── experiments.py    reproducible benchmarks
├── models.py         checksum-pinned weight download
├── face/             detector · aligner · embedder · matcher · enrollment · backends
├── protection/       watermark · fingerprint · adversarial
├── detection/        deepfake · deepfake_backends · watermark_detector · manipulation
├── video/            sampler · tracker · processor
├── provenance/       hash_chain · metadata · c2pa_adapter
├── risk/             features · scorer · calibration
├── storage/          repositories for identities, assets, evidence, provenance
├── pipeline/         protection_pipeline · analysis_pipeline
└── api/              app · routes
```

Every replaceable model sits behind an abstract interface resolved through a
registry, so no phase is welded to one vendor.

---

## Installation

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,face,video,api,experiments]"
./bin/deepshield download-models --insightface
```

Model weights are not committed to this repository — the ONNX detector shards
are 330 MB each, past GitHub's file limit — so the download step above is
required, not optional. `download-models` fetches the checksum-pinned YuNet and
SFace weights plus the InsightFace pack; the published deepfake detectors are
exported separately with `python scripts/fetch_deepfake_detector.py --model
<name>`, which writes `models/deepfake_<name>.onnx` alongside the metadata JSON
that is committed.

The default embedder is ArcFace, which needs the InsightFace model pack. Without
it the pipeline reports a missing model and names the command to fetch it; set
`face.embedder.backend: opencv_sface` to run on the 37 MB fallback instead.

Extras: `face` (OpenCV, ONNX Runtime, InsightFace), `video` (OpenCV, PyAV),
`api` (FastAPI, SQLAlchemy), `experiments` (pandas, scikit-learn, matplotlib),
`torch` (for future learned watermarking). `deepshield doctor` reports what is
present.

### Use `./bin/deepshield`

`bin/deepshield` sets `PYTHONPATH` and invokes `python -m deepshield`. Prefer it
over the console script: on macOS with iCloud- or OneDrive-synced folders, the
`.pth` file written by `pip install -e` is intermittently ignored during
interpreter start-up, and the console script then fails with
`ModuleNotFoundError: No module named 'deepshield'`. The launcher and the
Makefile targets are unaffected. Keeping the repository on an ASCII path outside
a synced folder avoids the problem entirely.

---

## Quick Start

```bash
./bin/deepshield download-models                                  # fetch weights
./bin/deepshield enroll ./my_photos --user-id alice               # 3–10 photos
./bin/deepshield protect photo.jpg --user-id alice --distribution-id instagram
./bin/deepshield analyze-image suspect.jpg --user-id alice --save
./bin/deepshield analyze-video suspect.mp4 --user-id alice
./bin/deepshield report <analysis-id>
```

Add `--json` to any command for machine-readable output.

---

## CLI

| Command | What it does |
|---|---|
| `info` | configuration, backends, threshold calibration state |
| `doctor` | optional dependencies and model weight presence |
| `download-models` | fetch checksum-pinned weights (`--insightface` for ArcFace) |
| `enroll <dir> --user-id <id>` | build an identity template (`--min-images` to override) |
| `identities` | list enrolled identities |
| `protect <img> --user-id <id>` | watermark, fingerprint, register, record provenance |
| `analyze-image <img>` | full gated analysis, `--save` to store |
| `analyze-video <vid>` | sample, track, gate, aggregate |
| `watermark-detect <img>` | recover a code and resolve it to an asset |
| `fingerprint <img>` | hashes plus closest registered assets |
| `provenance <asset-id>` | lineage of a registered asset |
| `report <analysis-id>` | render a stored evidence record |
| `robustness-test <images>` | run a benchmark (`--experiment face\|watermark\|combined`) |
| `serve` | run the REST API |

Exit codes: `0` success, `1` runtime error, `2` usage error, `3` not implemented,
`4` a required model or dependency is unavailable.

---

## API

`./bin/deepshield serve`, then OpenAPI docs at `/docs`.

```text
POST /identity/enroll      POST /analyze/image     POST /watermark/detect
POST /protect/image        POST /analyze/video     GET  /analysis/{id}
GET  /identity             GET  /health            GET  /limitations
```

Domain errors map to meaningful statuses: `404` unknown identity or analysis,
`415` undecodable media, `422` enrollment quality failure, `503` missing model.

---

## Models

| Component | Default | Also supported |
|---|---|---|
| Face detector | `opencv_yunet` (YuNet ONNX, 232 KB) | `insightface` (SCRFD), `mock` |
| Face aligner | `landmark` (5-point similarity transform) | `simple_crop`, `mock` |
| Face embedder | `insightface` (ArcFace 512-d) with flip-TTA | `opencv_sface` (SFace 128-d, 37 MB), `onnx_arcface` (bring your own), `mock` |
| Deepfake detector | `spectral` (training-free heuristic) | `onnx` (bring your own; three published models measured at chance), `blending` (fitted, also at chance), `mock` |
| Watermark | `dct` (tiled DCT, CRC-32, grid resynchronisation) | `mock` |

Weights are never committed. They are pinned by URL and SHA-256 in
`src/deepshield/models.py` and verified on download. Model name and version are
recorded on every result, so a past analysis stays reproducible and an accidental
cross-model comparison is detectable.

ArcFace is the default because it was measured to be strictly better; see
**Precision** below. SFace remains available as a 37 MB fallback that needs no
extra Python package.

**Detector comparison.** YuNet is the default despite SCRFD being the stronger
model in general, because it was measured to be better *here*:

| Detector | clean | crop 20% | blur σ=3 | screenshot |
|---|---|---|---|---|
| YuNet | 0/170 missed | **0/170 missed** | 0/170 missed | 0/170 missed |
| SCRFD (InsightFace) | 1/170 missed | **167/170 missed** | 1/170 missed | 1/170 missed |

A detector that misses 98% of cropped probes turns every crop into a
no-evidence result. Configuration follows the measurement, not the reputation.

---

## Risk Score

The risk engine fuses signals into one explainable assessment. Three rules make
it defensible rather than merely numeric:

**Missing is not zero.** A signal that could not be computed stays `None`. The
weights of the signals that *were* computed are renormalised, so a watermark that
could not be checked lowers coverage, not risk.

**Positive evidence only, for attribution signals.** An attacker can strip a
watermark, so only a positive detection contributes. The same applies to
perceptual fingerprints: two unrelated images agree on roughly half their hash
bits by chance, so similarity below the evidence threshold is reported without
being scored.

**Uncalibrated signals are reported but not scored.** A detector whose threshold
has never been fitted produces a number with no defined operating point.
Including it would launder a guess into a percentage.

Real output for an unrelated person, showing all three rules at once:

```text
risk score: 19 / 100  (LOW)
  face_similarity: 0.187 (weight 100%, contributes 18.7 points)
  watermark: not detected, treated as neutral rather than exculpatory

limitations:
  - Face similarity does not prove that an image was used as training data.
  - Only 40% of the intended signal weight was available; the score is computed
    over the signals that were present.
  - The closest registered asset matched at 0.562 perceptual similarity, below
    the 0.812 evidence threshold. Unrelated images score around 0.5 by chance,
    so this was reported but not counted toward the risk score.
  - The synthetic-media detector was not run: no face passed the identity
    candidate threshold, so there was nothing to attribute to this user.
```

---

## Precision

Identity claims are the output a user acts on, so the pipeline is tuned for
precision rather than accuracy. Three things make that measurable rather than
aspirational: an evaluation set large enough to separate configurations, a
protocol that matches what the system actually does, and thresholds fitted on
the result.

### Protocol

```bash
python scripts/build_evaluation_set.py --identities 30 --per-identity 8
python scripts/evaluate_face_pipeline.py --embedders opencv_sface insightface \
    --tta 0 1 --ensemble --target-precision 0.99
python scripts/calibrate_thresholds.py --write
```

The evaluation set is 170 photos across 30 identities from Labeled Faces in the
Wild, filtered to images with exactly one clear face. Matching is evaluated
leave-one-out: each photo becomes a probe, the identity's remaining photos form
its gallery, and every other identity forms an impostor gallery. **Probes are
degraded, galleries are not** — enrollment photos are chosen by the user, while
suspect content arrives re-encoded, cropped or screenshotted. That asymmetry is
also what makes the comparison meaningful at all: on clean frontal portraits
every backend here scores a perfect AUC and nothing can be told apart.

### Backend comparison

Pooled over five probe conditions, 850 genuine and 24 650 impostor comparisons:

| Embedder | pooled EER | genuine min | impostor max | separable |
|---|---|---|---|---|
| SFace 128-d | 0.0012 | 0.304 | 0.488 | **no — distributions overlap** |
| SFace + flip-TTA | 0.0010 | 0.311 | 0.488 | no |
| **ArcFace 512-d + flip-TTA** | **0.0000** | **0.440** | **0.332** | **yes, gap 0.107** |

Per condition, mean recall at precision ≥ 0.99:

| Probe condition | SFace | SFace+TTA | ArcFace | ArcFace+TTA |
|---|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 | 1.000 |
| JPEG q30 | 1.000 | 1.000 | 1.000 | 1.000 |
| crop 20% | 0.994 | 1.000 | 1.000 | 1.000 |
| Gaussian blur σ=3 | 0.994 | 0.994 | 1.000 | 1.000 |
| screenshot (0.6×, q60) | 1.000 | 1.000 | 1.000 | 1.000 |
| **mean** | 0.998 | 0.999 | **1.000** | **1.000** |

Three conclusions, all acted on:

- **ArcFace replaced SFace as the default.** It is the only configuration whose
  genuine and impostor distributions do not overlap under degradation.
- **Flip-TTA is enabled by default.** It costs one extra forward pass and closed
  the crop-20% gap for SFace. It changed nothing measurable for ArcFace, which
  is why it is a knob rather than a claim.
- **Ensembling ArcFace with SFace was evaluated and rejected.** It matched
  ArcFace alone at double the cost. Fusion remains available
  (`face.embedder.ensemble`) for cases where a second model actually helps.

Aggregation strategies were compared on the same data and are statistically
indistinguishable here (mean EER 0.00040 for `centroid`, 0.00041 for `max` and
`topk_mean`), so `max` was kept rather than changed without evidence.

One condition defeats the pipeline outright: **downscaling a 250-pixel image to
25% produced no detections at all**, so it yields no score rather than a wrong
one. Detection failure and misidentification are reported separately throughout,
because they have different causes and different fixes.

### Threshold placement

With separable distributions, both boundaries are placed relative to the
midpoint of the gap rather than on either distribution's edge:

```text
impostor max 0.3323 ──┬── 0.3591 candidate ── 0.3860 midpoint ── 0.4128 high ──┬── 0.4397 genuine min
                      └──────────────── gap 0.1074 ───────────────────────────┘
```

Putting the boundary at the lowest genuine score scores perfectly on the
measured data and leaves no headroom at all — the next slightly harder genuine
pair falls straight through it. Each threshold is offset from the midpoint by a
quarter of the gap, so `candidate` errs toward catching things and
`high_confidence` errs toward being sure. Both currently reach precision 1.000
and recall 1.000 with zero false positives across 25 500 comparisons.

### Two-tier identity decisions

Clearing the candidate threshold justifies spending an expensive detector on a
face. It does **not** justify telling a user their face was found. Only a
high-confidence decision populates `matched_user_id`; the four states are
reported distinctly:

| decision | meaning | reported as |
|---|---|---|
| `no_match` | below the candidate threshold | no identity conclusion |
| `candidate` | above candidate, below high confidence | "worth reviewing, not a match" |
| `ambiguous` | best and runner-up identities within `min_margin` | "too small a gap to identify either" |
| `high_confidence` | above threshold, unambiguous | the identity is named |

Two guards support this. The **margin** between the best and runner-up identity
demotes near-ties: a probe scoring 0.62 against one identity and 0.61 against
another has identified nobody, however high the absolute number looks. Observed
correct-identification margins never fell below 0.431, so the 0.05 guard costs
nothing on this data and only fires on genuine ambiguity. It is inert when a
single identity is enrolled, which is the common personal case.

The **probe quality** score combines face resolution against the embedder's
112-pixel input and crop sharpness against the measured median of clean crops,
taking the worse of the two. It can raise the decision threshold in proportion
to how unreliable a probe is. Its cost was measured before switching it on:

| condition | median quality | recall without guard | recall with 0.10 guard |
|---|---|---|---|
| clean / JPEG / screenshot | 0.83 – 1.00 | 1.000 | 1.000 |
| Gaussian blur σ=3 | 0.036 | 1.000 | **0.894** |

On this pipeline the guard only costs recall and prevents no false positive,
because ArcFace already separates perfectly under blur. **It is therefore
disabled by default** (`low_quality_penalty: 0.0`) and kept as a documented knob
for weaker embedders, where blurred probes do overlap the impostor
distribution. The quality score is still computed and reported on every face.

### Effect on the risk score

Switching the default embedder moved an unrelated person's score from 0.187 to
0.056, which propagates straight into the risk output:

```text
before (SFace)   tesla_1.jpg vs enrolled 'curie'   similarity 0.187   risk 19 LOW
after  (ArcFace) tesla_1.jpg vs enrolled 'curie'   similarity 0.056   risk  6 LOW
```

### What these numbers do not mean

LFW is drawn from news photography and is skewed by demographic and by pose;
its own authors say so. Comparisons within one identity are not independent, so
850 genuine comparisons do not carry 850 comparisons' worth of evidence. These
results rank configurations and place provisional boundaries on this population.
They are not an accuracy claim for any particular user, and the calibration
report records the identity and photo counts behind every value so that the
limit travels with the number.

### Watermark resynchronisation

The DCT watermark originally lost every geometric attack: cropping moves and
rescales the 8x8 block grid the decoder depends on. Three changes recovered it.

**Tiled bit layout.** Bits were assigned in raster order, which ties the mapping
to the image width — and cropping changes the width, scrambling every bit even
when the grid is otherwise found. Bits are now assigned by position within a
repeating 8x8 tile, which is width-independent.

**Grid search.** The decoder tries seven magnifications and all sixty-four
sub-block pixel offsets, scoring each on a subsample of blocks, then all
sixty-four tile phases on the best candidates.

**A wider checksum, and a bounded search.** Every extra candidate is another
chance for a checksum to pass on noise. CRC-8 over the original search produced
**37 false detections in 40 unmarked images**; CRC-16 fixed that but could not
fund a grid search. CRC-32 can, and the checksum is verified with a vectorised
table-driven implementation so the search takes about 0.3 s instead of minutes.

The search still had to be capped twice, both times on measurement rather than
on theory:

- **Bit flipping is disabled inside the grid search.** It recovered the same 57
  of 60 cropped images while adding sixteen thousand candidate messages per
  image. One more recovery is not worth one wrong channel attribution.
- **Every valid candidate must name the same code.** Candidates from a wrong
  tile phase are permutations of the real message, not random bits, so the
  checksum passes far more often than the uniform one-in-four-billion estimate
  predicts. A 2 550-measurement benchmark produced exactly one accidental pass.
  Collecting all validating phases and rejecting when two disagree removed it:
  the guard fired twice on 170 cropped images and turned both into honest
  non-detections, at a cost of one recovered image.

Result on the 512-pixel synthetic reference at strength 0.16 (PSNR 35.7 dB,
SSIM 0.927):

| transformation | before | after |
|---|---|---|
| JPEG q90/q70, WebP, blur, noise, brightness, contrast, screenshot | recovered | recovered |
| resize 75% / 50% | recovered | recovered |
| **JPEG q50** | lost | **recovered** |
| **crop 10% / 20% / 30%** | lost | **recovered** |
| rotate 5° | lost | lost |
| resize 25% | lost | lost |
| **total** | **10 / 17** | **15 / 17** |
| **wrong codes** | 0 | **0** |
| **false positives on unmarked** | 0 / 30 | **0 / 30** |

Resolution matters, and the honest number is the one from real photographs. On
170 LFW images at 250 pixels — a quarter of the block budget of the synthetic
reference — nine transformations are recovered perfectly, cropping is recovered
at 10% and 20% but not 30%, and resize 50% degrades to roughly half:

| transformation | detected | wrong codes |
|---|---|---|
| JPEG q90/q70, WebP, blur, noise, brightness, contrast, screenshot, resize 75% | 170/170 | 0 |
| crop 10% | 167/170 | **0** |
| crop 20% | 164/170 | **0** |
| resize 50% | ~88/170 | 0 |
| JPEG q50, resize 25%, crop 30%, rotate 5° | 0/170 | 0 |

Smaller images carry fewer blocks, so each message bit gets fewer votes and the
resynchronisation search has less signal to lock onto. Embedding cost on this
set is PSNR 36.2 dB and SSIM 0.909.

One gate had to be removed to get there. Soft decoding was gated on vote
agreement, which collapses under downscaling even when the majority of bits are
still correct — the gate was rejecting decodes that were two bits from valid. A
32-bit checksum is a better guard than a heuristic that rejects correct answers.

### The deepfake signal: a negative result

The synthetic-media score was the one signal computed but never scored, because
no threshold had been fitted. Fitting one needs labelled data, so this session
built some and tried:

```bash
python scripts/build_manipulation_set.py      # 169 real + 169 face swaps
python scripts/train_deepfake_detector.py --write
```

The manipulation set is graphics-based face swapping: a donor face warped onto
the target over a Delaunay mesh of 106 landmarks, colour-matched, and composited
with Poisson blending. Donors are picked by pose similarity, because warping a
three-quarter view onto a frontal face produces something no attacker would
publish and no detector should be credited for catching.

The detector is eleven hand-crafted blending-artefact features — noise residual,
sharpness, JPEG blockiness, spectral slope, colour statistics and seam edge
energy, each as a ratio between the middle and the border of the face crop —
under a logistic regression, evaluated on **identity-disjoint** splits.

**It does not work.**

| split | ROC-AUC |
|---|---|
| held-out identities, clean | 0.579 |
| held-out identities, JPEG q50 | 0.536 |
| held-out identities, resize 50% | 0.545 |
| **in-sample** (can the features fit at all?) | **0.641** |
| best single feature (interior noise) | 0.601 |

The in-sample number is the informative one: at 0.64 the features cannot fit the
task even with the answers in front of them, so this is not overfitting or a
split artefact — the features are inadequate. The reason is mechanical. Poisson
blending exists precisely to remove the gradient discontinuity at the seam,
which is what most of these features measure.

Two things followed automatically, and both are the system behaving correctly:

- `train_deepfake_detector.py` **refused to write `calibrated: true`**, because
  the held-out AUC was below its 0.75 floor. The guard was written before the
  result was known.
- The risk engine therefore **still excludes the deepfake signal** from every
  score, and says so in the limitations of every report.

The `blending` backend ships anyway, with `usable: false` and its measured AUC
in the model file, and it repeats that AUC in the notes attached to every score
it produces. A detector at chance level is not evidence, and the code says so
rather than leaving a reader to assume otherwise.

This is the clearest confirmation of the project's premise available: signals
fail independently, and a system that had folded this one into a risk percentage
would have been reporting noise as forensics.

### Published detectors do not transfer either

Building a detector failed, so the next step was to stop building and go
shopping. Three published checkpoints were downloaded, exported to ONNX with
their own preprocessing baked into the graph, and run through this project's
existing `onnx` adapter on the same face crops the pipeline produces:

```bash
python scripts/fetch_deepfake_detector.py --model dima806
python scripts/evaluate_deepfake_detectors.py --write
```

| detector | HF downloads | ROC-AUC | recall at its own 0.5 point | **false positives on real photos** | verdict |
|---|---|---|---|---|---|
| `Wvolf/ViT_Deepfake_Detection` | 4 109 | 0.594 | 0.864 | **76.3%** | not usable |
| `dima806/deepfake_vs_real_image_detection` | 13 734 | 0.586 | 0.811 | **70.4%** | not usable |
| `prithivMLmods/Deep-Fake-Detector-Model` | 8 108 | **0.385** | 0.290 | 45.6% | not usable |

The AUC column is the polite way to read this. The right-hand column is the
real one: the two best-separating models call **seven out of ten genuine
photographs synthetic**. In a system whose entire purpose is to avoid
accusing someone falsely, a detector like that is worse than no detector — and
it would look like it was working, because it does catch most of the fakes.

The third model scores **below chance**, meaning it is anti-correlated with the
truth on this material. Its labels are also reversed relative to the other two
(`Fake` is class 0), which is exactly the wiring mistake that turns a working
detector into a confidently inverted one; the export script resolves the
synthetic class from the checkpoint's own label map and records which index it
chose, so that failure is caught rather than discovered later.

`--write` adopts the best qualifying detector automatically. It adopted nothing,
because none cleared the bar of AUC 0.75 with a false positive rate under 10%.

**Four independent attempts, one conclusion.** Hand-crafted blending features
(0.58), and three published detectors (0.59, 0.59, 0.39). The deepfake signal
stays out of the risk score, and the honest reading is that off-the-shelf
synthetic-media detection does not survive contact with material outside its
training distribution. That is not a gap in this project's engineering; it is
the finding, and it is the strongest possible argument for the multi-signal
design — the identity and attribution signals that *do* work carry the system
without it.

---

## Calibration

Thresholds are fitted on real data, not copied from a paper:

```bash
python scripts/fetch_sample_faces.py        # public-domain samples, git-ignored
python scripts/calibrate_thresholds.py --write
```

Measured with YuNet + landmark alignment + ArcFace + flip-TTA, pooled over five
probe conditions:

```text
pooled: 850 genuine, 24650 impostor
ROC-AUC 1.0000   EER 0.0000
genuine  min=0.4397 p1=0.4899 median=0.7482
impostor max=0.3323 p99=0.1833
placement: gap [0.3323, 0.4397] width 0.1074; midpoint -/+ 25% of the gap
candidate threshold       0.3591  (recall 1.0000, 0 false positives)
high-confidence threshold 0.4128  (precision 1.0000, recall 1.0000, 0 false positives)
```

Per condition, every one separates cleanly:

| condition | AUC | EER | genuine min | impostor max |
|---|---|---|---|---|
| clean | 1.0000 | 0.0000 | 0.566 | 0.279 |
| JPEG q30 | 1.0000 | 0.0000 | 0.553 | 0.288 |
| crop 20% | 1.0000 | 0.0000 | 0.544 | 0.287 |
| blur σ=3 | 1.0000 | 0.0000 | 0.440 | 0.332 |
| screenshot | 1.0000 | 0.0000 | 0.566 | 0.282 |

**This calibration is still provisional and the code says so.** Thirty
identities from one benchmark is not a deployment population, and comparisons
within an identity are not independent. The report records the identity and
photo counts, and `deepshield info` prints the file every threshold came from.
The deepfake threshold remains uncalibrated — there is no labelled deepfake data
here — so that signal is excluded from every risk score.

---

## Experiments

All benchmarks write CSV to `data/results/` with seed, git commit, Python
version, model versions and transformation parameters on every row.

### Face recognition robustness (RQ2)

`./bin/deepshield robustness-test data/test/faces --experiment face`
— 5 images × 15 transformations:

| transformation | mean similarity | min | no face |
|---|---|---|---|
| resize 75% | 0.994 | 0.982 | 0 |
| screenshot | 0.990 | 0.973 | 0 |
| JPEG q90 | 0.988 | 0.963 | 0 |
| resize 50% | 0.987 | 0.952 | 0 |
| JPEG q70 / q50 | 0.977 | 0.945 | 0 |
| rotate 5° | 0.959 | 0.932 | 0 |
| crop 10% | 0.951 | 0.922 | 0 |
| brightness ×1.2 | 0.949 | 0.912 | 0 |
| **crop 30%** | — | — | **5/5 undetected** |

Identity survives every common transformation well above the 0.436 candidate
threshold. It fails on heavy cropping — and fails by *detection*, not by
mis-identification, which is a different failure with a different fix.

### Watermark robustness (RQ1)

`./bin/deepshield robustness-test <images> --experiment watermark`. See
**Watermark resynchronisation** above for the full before/after table: 15 of 17
transformations survive, including all three crop levels and JPEG q50, with zero
wrong codes and zero detections across unmarked images. Rotation and a 4×
downscale remain unrecovered, and the decoder reports failure rather than
inventing a code for them.

### Adversarial cloaking (RQ2, RQ3) — a negative result

SPSA cloaking against SFace, measured against the transformation-only baseline:

| condition | white-box | cross-model (ArcFace) | JPEG-70 | SSIM |
|---|---|---|---|---|
| *baseline: JPEG-70 alone, no cloak* | *0.022* | — | — | — |
| *baseline: resize-50 alone, no cloak* | *0.046* | — | — | — |
| ε=0.03, 300 steps | 0.016 | 0.007 | 0.037 | 0.962 |
| ε=0.06, 300 steps | 0.080 | 0.027 | 0.034 | 0.873 |

**At perceptually acceptable budgets the cloak displaces the embedding less than
ordinary JPEG compression does**, and cross-model transfer is three times weaker
still. At ε=0.06 the cosine similarity is still 0.97 — recognition succeeds
trivially — and the image is visibly degraded. Cloaking is kept as research,
disabled by default, and is not presented as protection.

### Combined protection (RQ4)

`./bin/deepshield robustness-test data/test/faces --experiment combined`:

| condition | PSNR | SSIM | face similarity | watermark recovered |
|---|---|---|---|---|
| original | — | 1.000 | 1.000 | 0/5 |
| watermark only | 38.7 | 0.927 | 0.986 | 5/5 |
| adversarial only | 34.5 | 0.908 | 0.975 | 0/5 |
| both | 32.6 | 0.850 | 0.973 | 5/5 |

The two layers **do not destroy each other**: watermark recovery stays 5/5 under
cloaking. Their perceptual cost compounds (SSIM 0.927 → 0.850), and neither
meaningfully reduces face similarity.

### Evaluation policy

Accuracy alone is never reported. Face matching is judged on TPR, FPR, ROC-AUC,
EER, precision and recall; watermarking on detection rate, bit accuracy and false
attribution rate; adversarial protection on white-box, cross-model and
post-transformation displacement, reported separately. Image quality is PSNR and
SSIM.

### Research questions

- **RQ1** Watermark survives social-media transformations at acceptable quality?
  **Yes for compression, rescaling and cropping** after grid resynchronisation
  (15/17); rotation and heavy downscaling still defeat it.
- **RQ2** Cloaking survives JPEG, resize, crop, screenshot? **No — it is weaker
  than the transformations themselves.**
- **RQ3** Cloaking transfers across face models? **Poorly** — 0.027 vs 0.080.
- **RQ4** Do watermarking and cloaking interfere? **No**, but quality cost adds.
- **RQ5** Does multi-signal fusion reduce false positives? Implemented and
  visible in the risk output. Partly answered by RQ8: a system relying on the
  deepfake signal alone would be reporting a chance-level score as forensics.
- **RQ8** Can a hand-crafted blending detector be calibrated on locally
  generated face swaps? **No** — 0.579 held-out and 0.641 in-sample AUC.
- **RQ9** Do published deepfake detectors transfer to this material? **No** —
  three checkpoints scored 0.59, 0.59 and 0.39 AUC, and the two best call over
  70% of real photographs synthetic. The signal stays out of the risk score.
- **RQ6** Sampling frequency vs accuracy and cost? **Measured**: on a two-person
  clip the identity is recovered at every rate from 0.5 to 10 fps, while cost
  scales with the frame count. 1 fps recovers it at 0.942 similarity in 0.29 s;
  10 fps reaches 0.975 for 0.88 s. The extra frames buy accuracy that the
  decision threshold does not need, which is why 1 fps is the default.
- **RQ7** Tracking and representative frames cut cost without hurting recall?
  Implemented; one embedding per track instead of per frame.

---

## Limitations

1. **No training-data attribution.** Out of scope, by design.
2. **Calibration is provisional.** 30 identities from LFW, a benchmark skewed by
   demographic and pose, with non-independent within-identity comparisons. Refit
   on data resembling the deployment population before any real use.
3. **The deepfake signal is uncalibrated and excluded from the risk score**, and
   four attempts failed to change that: a hand-crafted blending detector (0.58
   AUC) and three published checkpoints (0.59, 0.59, 0.39), the best of which
   flags 70% of genuine photographs as synthetic. The `onnx` adapter is wired,
   tested and ready; what is missing is a model that generalises, which needs a
   real labelled corpus behind a data-use agreement.
4. **Watermarks die on rotation and heavy downscaling**, and on small images
   they die sooner: crop 30% and JPEG q50 survive on a 512-pixel image but not
   on a 250-pixel one. Cropping recovery costs about 0.3 s per undetected image.
5. **Cloaking does not work** at acceptable quality budgets — measured above.
6. **Face recognition fails when detection fails** — heavy crops and a 4×
   downscale of an already small image produce no detection at all. It also
   inherits its training set's demographic bias, which LFW is the wrong set to
   measure and this project does not attempt to.
7. **Video tracking is heuristic.** IoU plus a colour-histogram appearance check
   separates people across a cut; fast motion and crossing faces still defeat it.
   It is tuned to err toward over-splitting, because a merged track silently
   drops a person from the report.
8. **C2PA is a declared stub.** `verify()` returns `supported: false` rather than
   a false negative, so callers can distinguish "no credential" from "cannot
   check".
9. **No web monitoring.** Input is what the user submits.

---

## Privacy

Face embeddings are biometric identifiers and are handled as sensitive data:

- Identity metadata and embedding vectors are written to **separate directories**,
  so the biometric store can be permissioned, encrypted or deleted on its own.
- `data/` is git-ignored; no face image or embedding is committed.
- Logs never contain a full embedding. `safe_embedding_repr` truncates to a
  preview plus a non-reversible digest, and `EmbeddingRedactionFilter` drops any
  record flagged as carrying a raw vector.
- Watermark payloads carry opaque identifiers only — never a name, e-mail or
  handle. The database maps codes to metadata.
- `to_dict()` on every biometric type omits the raw vector by construction.
- User ids are sanitised before use as filenames, so no identifier can escape its
  directory.

---

## Development

```bash
make test    # pytest
make lint    # ruff and mypy
make info    # configuration and backends
make clean
```

Current state: **478 tests passing, ruff clean, mypy clean on 51 source files.**

Code is production-style: type hints, docstrings, explicit exception types,
configuration separation, logging, and no inline comments. Every phase followed
the same loop — implement, unit test, sample test, compute metrics, record
failures, update this README.

### Roadmap status

| Phase | Scope | Difficulty | Status |
|---|---|---|---|
| 0 | Skeleton, config, logging, CLI, interfaces | EASY | done |
| 1–2 | Face detection, alignment, embedding | EASY | done |
| 3–4 | Enrollment, similarity and aggregation | EASY | done |
| 5 | Image analysis pipeline | EASY | done |
| 6 | Video sampling, tracking, representative frames | MEDIUM | done |
| 7 | Deepfake detector adapter | MEDIUM | done; both backends measured at chance, signal excluded |
| 8 | Image fingerprint | EASY | done |
| 9 | Watermark (tiled DCT + resynchronisation) | MEDIUM | done |
| 10 | Risk engine and calibration | MEDIUM/HARD | done (provisional) |
| 11 | Protection pipeline | MEDIUM | done |
| 12 | Adversarial protection research | HARD | done — negative result |
| 13 | Robustness benchmarks | MEDIUM | done |
| 14 | REST API | MEDIUM | done |
| 15 | Provenance; C2PA adapter | MEDIUM | done; C2PA stubbed |
| 16 | Web monitoring connectors | HARD | out of MVP |

Next, in order of value:

1. **A deepfake detector that generalises.** Measured four times now — one
   hand-built, three published — and all sit at or below chance on this
   material. The adapter, the export path, the survey harness and the adoption
   guard are all built and tested; `evaluate_deepfake_detectors.py --write`
   will wire in the first model that clears AUC 0.75 at under 10% false
   positives. What is missing is that model, which needs training on a real
   labelled corpus such as FaceForensics++ or Celeb-DF. Both require a signed
   data-use agreement, which is the one step this repository cannot take on its
   own.
2. **Rotation-invariant watermark synchronisation**, the one geometric attack the
   grid search does not cover.
3. **A calibration set resembling a real deployment population** rather than a
   news-photography benchmark.

Deliberately excluded from the MVP: internet-wide crawling, full C2PA
infrastructure, training-data attribution, membership inference as a product
verdict, dataset inference, machine unlearning, training-set poisoning, and
universal generator or watermark-attack coverage.
