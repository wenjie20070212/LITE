# Training protocol (all reported experiments)

## Data

| Dataset | Classes | Train | Val | Test | Pair definition |
|---|---:|---:|---:|---:|---|
| Soybean Disease | 8 | 6,754 | 964 | 1,930 | image + caption + cached feature |
| PlantDoc | 27 | 2,083 | 233 | 236 | image + caption + cached feature |

- Split seed: 42, stratified per class (see `data/splits/README.md`).
- Soybean: all text-paired samples pooled and split 70/10/20.
- PlantDoc: official `test` folder used as TEST (236 images); the remaining
  text-paired images split 90/10 into TRAIN/VAL.
- Class order: `sorted()` on directory names; label map in the manifests.

## Inputs

| Item | Setting |
|---|---|
| Image size | resize 256 → RandomResizedCrop 224 (train) / CenterCrop 224 (val, test) |
| Augmentation | random horizontal flip (train only) |
| Normalization | ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] |
| Text feature | cached `(1, L, 768)` per image, per-token L2-normalized |

### Text feature pipeline

The released checkpoints were trained with exactly the released English
pipeline (documented in the main `README.md` and shipped in
`data/captions/`): GLM-4.6V English captions, encoded by
`bert-base-uncased` with `max_length=128`, `padding="max_length"`,
`truncation=True`, per-token L2 normalization, `(1, 128, 768)` float32.
The teacher pools token vectors by plain mean **without an attention mask**
(PAD positions participate) — reported for transparency.

Caption-generation sampling (stage 1): temperature 0.3, output length capped
at 1,024 tokens (`text_pipeline/generate_captions.py`). Caption quality is
audited by `scripts/audit_captions.py` (see `data/captions/README.md`).

## Optimization (identical for teacher and student)

| Item | Setting |
|---|---|
| Optimizer | Adam, lr 1e-3, weight decay 1e-4 |
| Batch size | 64 |
| Epochs | 200 |
| LR schedule | StepLR, step 50, gamma 0.5 |
| Grad clip | 1.0 (global norm) |
| Regularization | no MixUp, no label smoothing, no early stop |
| Seeds | 7 / 42 / 123 (teacher: seed 7) |
| Model selection | highest **validation** accuracy; TEST evaluated exactly once afterwards |

## Teacher

`ResTextTeacher` = ShuffleNetV2-x2.0 image branch + text-feature branch
(768 → 256 projection, token mean pooling) + 4-head cross-attention
(image query over text tokens) + concat fusion MLP; trained with plain
cross-entropy. Loss weights for distillation refer to this teacher's features.

## Student (main model) and distillation loss

`ResTextStudent` = ShuffleNetV2-x1.0 + width-matched three-layer MLP text
simulator (256 → 620 → 620 → 256, ~0.706M simulator parameters, capacity
matched to the multi-scale simulator) + lightweight fusion + classifier.

Loss: `total = alpha·hard + beta·soft + gamma·feature` with

| Term | Weight | Definition |
|---|---:|---|
| hard | α = 0.7 | cross-entropy on true labels |
| soft | β = 0.1 | KL divergence at temperature **T = 8**, scaled by T² |
| feature | γ = 0.05 | three-way alignment, each way `MSE + (1 − cosine)`, weights img 0.3 / text 0.2 / fused 0.5, normalized by the weight sum (1.0) |

`lambda_T = 0` ablation: the text-alignment weight is set to 0 **and** the
feature denominator is fixed at 1.0, so the remaining terms keep exactly their
base scale (no re-normalization). Architecture, optimizer, schedule and seeds
are unchanged. Run it with `scripts/train.py --lambda-text 0`.

## Significance analysis

The per-sample TEST predictions of the released main model (three seeds) are
in `data/predictions/`; `scripts/paired_test.py` computes the paired accuracy
difference, its 95% bootstrap CI (**100,000 resamples**, fixed seed), a
two-sided bootstrap p-value, and exact McNemar p-value. Differences whose CI
contains 0 are reported as not statistically confirmed.

## Reproducibility

- Seed everything (`random`, `numpy`, `torch`, `torch.cuda`) with the run seed.
- Deterministic dataset order: class list `sorted()`, files iterated in sorted
  order; train loader shuffles, val/test loaders do not.
- No validation-set training: TEST is touched only by `scripts/eval.py` after
  the checkpoint is finalized.
