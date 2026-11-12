# LITE — Main-Model Release

This package accompanies the revised manuscript on **LITE (Learning To
Imagine Text Embeddings)**: a multimodal-to-unimodal knowledge distillation
framework for crop disease classification. A multimodal teacher
(ShuffleNetV2-x2.0 image branch + BERT text-feature branch + cross-attention)
is distilled into an image-only student (ShuffleNetV2-x1.0) whose internal
text simulator is supervised to mimic the teacher's text representation, so
that the deployed student needs no text input at inference time.


## Repository layout

```
models/                 model definitions
  distillation.py         teacher (ResTextTeacher), student (ResTextStudent,
                          width-matched MLP simulator), loss
                          (MultiLevelDistillationLoss: img/text/fused alignment)
utils/
  img_process.py          dataset loading (image + cached feature pairs),
                          augmentation, evaluation helpers
text_pipeline/           text feature pipeline (English)
  prompt_caption_en.txt   caption-generation prompt (class-agnostic)
  generate_captions.py    stage 1: image -> caption (.txt) via a GLM-4V-family VLM
  encode_text_features.py stage 2: caption (.txt) -> (1,128,768) .npy via
                          bert-base-uncased
scripts/
  make_splits.py          reproduce the released train/val/test splits
  verify_splits.py        check (image, caption, feature) pairing integrity
  train.py                train teacher / student (--lambda-text 0 = ablation)
  eval.py                 single test-set evaluation
  paired_test.py          paired bootstrap + McNemar from two prediction files
  audit_captions.py       caption leakage audit over the released captions
configs/protocol.md      full training protocol
data/splits/             split definitions (CSV lists + JSON manifests) and
                         the excluded/missing-sample documentation
data/captions/           README + leakage-audit results; the caption .txt files
                         themselves are a direct download (Baidu Pan — see the
                         "Released captions" note below) — unpack them as
                         <dataset>/{train,val,test}/<class>/<stem>.txt here
data/predictions/        per-sample TEST predictions of the released main
                         model (three seeds) — input for paired comparisons
checkpoints/             released main model, one checkpoint per seed
  soybean/mlp_matched_seed{7,42,123}/best.pth
  plantdoc/mlp_matched_seed{7,42,123}/best.pth
```


## Statistical significance analysis (paired comparisons)

`data/predictions/<dataset>/mlp_matched_seed<S>.csv` stores the per-sample
TEST predictions (`true,pred,correct`) of the released main model (seeds
7 / 42 / 123), evaluated on the same split in the same deterministic order, so
rows align across files and paired tests are valid. `scripts/paired_test.py`
verifies alignment, then reports the paired accuracy difference, its **95%
bootstrap CI** (100,000 resamples), a two-sided bootstrap p-value, and exact
**McNemar** p-value:

```bash
python scripts/paired_test.py \
    --a data/predictions/soybean/mlp_matched_seed7.csv \
    --b data/predictions/soybean/mlp_matched_seed42.csv
```

If a difference's CI contains 0, it is not statistically confirmed and should
be reported as such. Predictions of comparison runs are produced by the same
`scripts/eval.py` (same split, same deterministic order) and can be dropped
into `data/predictions/` as the second side of any paired test; see
`data/predictions/README.md`.

Images can be obtained from the original dataset sources; captions and cached
features are regenerated with the pipeline below.

# Text feature pipeline

## Stage 1 — caption generation (`text_pipeline/generate_captions.py`)

Images under `Image/{train,val,test}/<class>/` that lack a caption are sent to
a GLM-4V-family vision-language model (`--model glm-4.6v`,
endpoint `https://open.bigmodel.cn/api/paas/v4`) together with the fixed prompt
in `text_pipeline/prompt_caption_en.txt`. The prompt asks for ONE neutral
paragraph covering: (1) overall observation (healthy vs. diseased; spot-like /
patchy / diffuse distribution), (2) local features (lesion shape, size in mm,
margin, surface), and (3) color and texture (dominant color, center-to-margin
gradient, powdery / fuzzy / sticky / cracked texture), while explicitly
forbidding plant or disease names.

```bash
export ZHIPU_API_KEY=...          # key from the environment only, never a file
python text_pipeline/generate_captions.py --dataset-root <DS_ROOT> --dry-run
python text_pipeline/generate_captions.py --dataset-root <DS_ROOT> \
    --model glm-4.6v
# -> Text/<split>/<class>/<stem>.txt
```

Safety notes: the endpoint is validated (https + host allow-list + resolved-IP
public check; SSRF/DNS-rebinding protection), redirects are disabled, 429
responses honor `Retry-After`, existing captions are skipped (resumable), and
each call is appended to a CSV log.

> **Released captions.** Captions for ALL splits are provided as a direct
> download: [Baidu Pan](https://pan.baidu.com/s/1JLgWZ26_T-YamUoeGsdsQw?pwd=w3dk)
> (access code `w3dk`). Unpack them under `data/captions/` — the same directory
> documents per-split counts, placement instructions, and the reproducible
> leakage audit (`data/captions/audit/`).

## Stage 2 — feature encoding (`text_pipeline/encode_text_features.py`)

Captions are encoded with **`bert-base-uncased`** (`google-bert/bert-base-uncased`)
into token-level features and cached as `.npy` files:

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('google-bert/bert-base-uncased', \
local_dir='/path/to/bert-base-uncased', \
allow_patterns=['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','model.safetensors'])"
python text_pipeline/encode_text_features.py --dataset-root <DS_ROOT> \
    --bert-dir /path/to/bert-base-uncased
# -> Textfeature/<split>/<class>/<stem>.npy
```

Encoding specification (fixed):

| Item | Setting |
|---|---|
| Tokenizer / model | `bert-base-uncased`, 12 layers, hidden 768 |
| `max_length` | **128** (model limit 512; captions ≈60 words ≈80–90 BPE tokens) |
| Padding / truncation | `padding="max_length"`, `truncation=True` — every sequence is exactly 128 tokens; longer captions are hard-truncated |
| Output | `last_hidden_state` per token, **each token vector L2-normalized** (PAD positions included) |
| File format | float32, shape `(1, 128, 768)` (leading batch dim kept for the dataloader) |
| Downstream pooling | the teacher averages token vectors **without an attention mask**, i.e. `[PAD]` positions participate in the mean — an implementation detail kept identical across all experiments and reported here for transparency |

## Datasets and splits

| Dataset | Classes | Images | Text-paired | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|
| Soybean Disease | 8 | 9,648 | 9,648 | 6,754 | 964 | 1,930 |
| PlantDoc | 27 | 2,552 | 2,552 | 2,083 | 233 | 236 (official test folder) |

- Split seed **42**, stratified per class; rules in `data/splits/README.md`.
- `*_split_lists.csv`: one row per image — `split, class, image, has_caption, has_feature`
- `*_split_manifest.json`: classes, label map, per-class and per-split counts.
- **PlantDoc source completeness**: the local copy was verified byte-level
  (git blob SHA-1) against the public reference `pratikkayal/PlantDoc-Dataset`.
  The official test folder (used as TEST, 236 images) is complete and
  byte-identical; 26 reference train images (~1.1%) are absent from the local
  copy — 24 across 9 of the 27 classes (each class retains train samples) and
  2 of a non-task 28th class. Full list:
  `data/splits/plantdoc_source_missing_images.csv`; details in
  `data/splits/README.md`.

## Environment

See `requirements.txt` (Python 3.10, torch 2.6.0+cu124, torchvision 0.21.0,
transformers 4.49.0). One GPU (≥8 GB) is sufficient for the student; the
teacher benefits from ≥16 GB.

## Data preparation workflow

```bash
# 0) download the released captions (Baidu Pan, see the "Released captions"
#    note below; details in data/captions/README.md) and unpack them under
#    data/captions/  (<dataset>/<split>/<class>/<stem>.txt)
# 1) build the three-way splits from your captioned copy of the datasets
python scripts/make_splits.py \
    --source-root <SRC> --output-root <OUT>
# 2) verify (image, caption, feature) pairing integrity
python scripts/verify_splits.py --dataset-root <OUT>/Soybean_disease
python scripts/verify_splits.py --dataset-root <OUT>/plant-doc-dataset
# 3) generate captions for images that lack them   (stage 1 above)
# 4) encode all captions into cached .npy features (stage 2 above)
```

`scripts/make_splits.py` reproduces the released split lists **exactly**
(verified: identical file membership for both datasets, all six splits).

## Reproducing an evaluation

```bash
# Soybean (8 classes)
python scripts/eval.py \
  --ckpt checkpoints/soybean/mlp_matched_seed7/best.pth \
  --task student --dataset-path <SOYBEAN_ROOT> --num-classes 8 \
  --out eval_soybean --name mlp_matched_seed7

# PlantDoc (27 classes)
python scripts/eval.py \
  --ckpt checkpoints/plantdoc/mlp_matched_seed7/best.pth \
  --task student --dataset-path <PLANTDOC_ROOT> --num-classes 27 \
  --out eval_plantdoc --name mlp_matched_seed7
```

`--out` receives `metrics.json` (overall accuracy) and `predictions.csv`
(per-sample `true,pred,correct`), which is the input format for the paired
bootstrap used in the paper's significance analysis.

## Reproducing training

```bash
# 1) train the multimodal teacher (needed before distilling students)
python scripts/train.py --task teacher \
  --dataset-path <ROOT> --num-classes 8 --seed 7 --out-dir runs/teacher_mm

# 2) train the main (width-matched) student, three seeds
python scripts/train.py --task student \
  --dataset-path <ROOT> --num-classes 8 --seed 7 \
  --teacher-checkpoint runs/teacher_mm/best.pth --out-dir runs/student_seed7
python scripts/train.py --task student \
  --dataset-path <ROOT> --num-classes 8 --seed 42 \
  --teacher-checkpoint runs/teacher_mm/best.pth --out-dir runs/student_seed42
python scripts/train.py --task student \
  --dataset-path <ROOT> --num-classes 8 --seed 123 \
  --teacher-checkpoint runs/teacher_mm/best.pth --out-dir runs/student_seed123

# 3) lambda_T = 0 ablation (text-alignment loss term removed; everything
#    else identical — architecture, optimizer, schedule, seeds)
python scripts/train.py --task student --lambda-text 0 \
  --dataset-path <ROOT> --num-classes 8 --seed 7 \
  --teacher-checkpoint runs/teacher_mm/best.pth --out-dir runs/student_notext_seed7
```

Full hyper-parameters: `configs/protocol.md`. The checkpoint selected by
validation accuracy is saved to `<out-dir>/best.pth`; the TEST split is only
evaluated once afterwards via `scripts/eval.py`.


## License

Code: MIT (see `LICENSE`). Split files: CC BY 4.0.

## Contact

Jianlin Qiu $^{1}$, Depeng Gao $^{1}$, and Wenjie Liu $^{2}$ (correspondence:
lwj2014@ntu.edu.cn)

$^{1}$ School of Yonyou Digital Intelligence, Nantong Institute of Technology,
Nantong 226002, China

$^{2}$ School of Transportation and Civil Engineering, Nantong University,
Nantong 226019, China
