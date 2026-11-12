# Per-sample TEST predictions (main model)

This directory contains the **per-sample TEST-set predictions of the released
main model** (width-matched student, `mlp_matched`, seeds 7 / 42 / 123): one
`true,pred,correct` row per test image. They are the verifiable evidence
behind the reported means and the input required for **paired significance
analysis** (paired bootstrap / McNemar) over the same test set.

## Files

| Dataset | Test size | Files |
|---|---:|---|
| `soybean/` | 1,930 | `mlp_matched_seed7.csv`, `mlp_matched_seed42.csv`, `mlp_matched_seed123.csv` |
| `plantdoc/` | 236 | same three seeds |

## Row alignment (why pairing is valid)

Rows follow the training/evaluation dataloader order: classes in `sorted()`
order, files in sorted order within each class — identical to the order of
`test` rows in `data/splits/<dataset>_split_lists.csv`. Every file was
produced by evaluating the selected checkpoint on the same split with the
same loader, so row *i* is the same test image in every file.
`scripts/paired_test.py` verifies this by checking that the `true` columns of
the two files being compared are identical before pairing.

## Running a paired comparison

```bash
# example: cross-seed stability of the main model (Soybean, seeds 7 vs 42)
python scripts/paired_test.py \
    --a data/predictions/soybean/mlp_matched_seed7.csv \
    --b data/predictions/soybean/mlp_matched_seed42.csv \
    --resamples 100000 --seed 20261108
```

Output: per-model accuracy, the paired accuracy difference, the **95%
bootstrap CI** of the difference, a two-sided bootstrap p-value, and exact
**McNemar** p-value, plus an explicit statement of whether the difference is
statistically confirmed.

Per-sample predictions of any additional comparison run are produced by the
same `scripts/eval.py` on the same split in the same deterministic order;
drop such a file into this directory and it is immediately usable as the
`--b` side of a paired test against the main model.
