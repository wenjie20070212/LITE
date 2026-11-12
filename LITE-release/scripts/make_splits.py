"""Reproduce the released train/val/test split lists from a captioned dataset.

Split rules (identical to the experiments; seed 42):
  Soybean  - no official split exists; ALL text-paired samples (source train+val
             pooled) are split stratified 70/10/20 into TRAIN/VAL/TEST.
  PlantDoc - official split adopted: the official test folder is used as TEST
             unchanged; the remaining (train-pool) images are split stratified
             90/10 into TRAIN/VAL.

Input layout (per dataset):
    <source>/<dataset>/Image/{train,val}/<class>/<image>.jpg
    <source>/<dataset>/Textfeature/{train,val}/<class>/<stem>.npy
Output layout:
    <output>/<dataset>/Image/{train,val,test}/<class>/<image>.jpg
    <output>/<dataset>/Textfeature/{train,val,test}/<class>/<stem>.npy
    <output>/<dataset>/split_manifest.json

Files are HARD-LINKED (os.link) - no extra disk space, originals untouched.
Only pairs with a cached .npy feature are included, so the reported dataset
sizes equal what the models actually see.
"""
import argparse
import json
import os
import random
from pathlib import Path

SEED = 42
IMG_EXTS = {'.jpg', '.jpeg', '.png'}
DATASETS = {'soybean': 'Soybean_disease', 'plantdoc': 'plant-doc-dataset'}


def _safe_child(base, *parts):
    base = Path(base).resolve()
    p = base.joinpath(*parts).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f'path escapes base: {parts}')
    return p


def _safe_out(base, name):
    if os.sep in name or '/' in name or '..' in name:
        raise ValueError(f'unsafe output name: {name}')
    p = (Path(base) / name).resolve()
    if not p.is_relative_to(Path(base).resolve()):
        raise ValueError(f'path escapes base: {name}')
    return p


def valid_pairs(dataset_root, mode):
    """{class: [(img_path, feat_path), ...]} for images that have a cached .npy feature."""
    img_dir = _safe_child(dataset_root, 'Image', mode)
    tf_dir = _safe_child(dataset_root, 'Textfeature', mode)
    out = {}
    for c in sorted(os.listdir(img_dir)):
        ci, cf = _safe_child(img_dir, c), _safe_child(tf_dir, c)
        if not ci.is_dir():
            continue
        pairs = []
        for name in sorted(os.listdir(ci)):
            if os.path.splitext(name)[1].lower() not in IMG_EXTS:
                continue
            feat = _safe_child(cf, os.path.splitext(name)[0] + '.npy')
            if feat.exists():
                pairs.append((str(ci / name), str(feat)))
        out[c] = pairs
    return out


def stratified(pairs_by_class, ratios, rng):
    """Assign each class's pairs to splits by ratios (train, val, test).
    Two-way splits (test ratio 0) assign the remainder to val so nothing can
    leak into test; rounding never creates negative or leaking buckets."""
    result = {'train': {}, 'val': {}, 'test': {}}
    has_test = ratios[2] > 0
    for c, pairs in pairs_by_class.items():
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n = len(pairs)
        a = int(round(n * ratios[0]))
        b = int(round(n * ratios[1]))
        if not has_test:
            if n >= 2:
                a = min(max(a, 1), n - 1)
            else:
                a = n
            b = n - a
        else:
            if n >= 3:
                a = min(max(a, 1), n - 2)
                b = min(max(b, 1), n - a - 1)
            else:
                a, b = n, 0
        labels = ['train'] * a + ['val'] * b + ['test'] * (n - a - b)
        for i, split in zip(idx, labels):
            result[split].setdefault(c, []).append(pairs[i])
    return result


def materialize(src_name, dst_root, split_map, rule):
    """Hard-link split files into place and write split_manifest.json."""
    classes = sorted({c for s in split_map.values() for c in s})
    for split in ('train', 'val', 'test'):
        for c in classes:
            _safe_child(dst_root, 'Image', split, c).mkdir(parents=True, exist_ok=True)
            _safe_child(dst_root, 'Textfeature', split, c).mkdir(parents=True, exist_ok=True)
    n_link = 0
    for split, by_class in split_map.items():
        for c, pairs in by_class.items():
            for img_path, feat_path in pairs:
                name = os.path.basename(img_path)
                stem = os.path.splitext(name)[0]
                for src, dst in (
                        (img_path, _safe_child(dst_root, 'Image', split, c, name)),
                        (feat_path, _safe_child(dst_root, 'Textfeature', split, c,
                                                stem + '.npy'))):
                    if os.path.exists(dst):
                        continue
                    os.link(src, dst)  # hard link: same filesystem, zero copy cost
                n_link += 1
    counts = {s: {c: len(split_map[s].get(c, [])) for c in classes}
              for s in ('train', 'val', 'test')}
    manifest = {
        'source': str(src_name),
        'seed': SEED,
        'rule': rule,
        'classes': classes,
        'label_map': {c: i for i, c in enumerate(classes)},
        'counts_per_class': counts,
        'totals': {s: sum(counts[s].values()) for s in counts},
        'linked_pairs': n_link,
    }
    _safe_out(dst_root, 'split_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--source-root', required=True,
                    help='root containing Soybean_disease/ and plant-doc-dataset/ '
                         'in source layout (Image/{train,val} + Textfeature)')
    ap.add_argument('--output-root', required=True,
                    help='destination root for the three-way splits')
    args = ap.parse_args()

    rng = random.Random(SEED)

    # ---------- PlantDoc: adopt official split ----------
    pd_src = _safe_child(args.source_root, DATASETS['plantdoc'])
    pd_train_pool = valid_pairs(pd_src, 'train')
    pd_official_test = valid_pairs(pd_src, 'val')
    split_map = stratified(pd_train_pool, (0.9, 0.1, 0.0), rng)
    # official test is used as TEST unchanged (no re-splitting, no leakage)
    for c, pairs in pd_official_test.items():
        split_map['test'].setdefault(c, []).extend(pairs)
    pd_manifest = materialize(
        DATASETS['plantdoc'], _safe_child(args.output_root, DATASETS['plantdoc']),
        split_map,
        'official test split adopted as TEST (untouched); official train pool '
        'stratified 90/10 into TRAIN/VAL; seed 42')
    print('PlantDoc:', pd_manifest['totals'])

    # ---------- Soybean: pool + stratified 70/10/20 ----------
    sb_src = _safe_child(args.source_root, DATASETS['soybean'])
    sb_train = valid_pairs(sb_src, 'train')
    sb_val = valid_pairs(sb_src, 'val')
    pooled = {}
    for c in sorted(set(sb_train) | set(sb_val)):
        pooled[c] = sb_train.get(c, []) + sb_val.get(c, [])
    split_map = stratified(pooled, (0.7, 0.1, 0.2), rng)
    sb_manifest = materialize(
        'Soybean_disease', _safe_child(args.output_root, 'Soybean_disease'), split_map,
        'no official split; all valid pairs (train+val pooled) stratified '
        '70/10/20 into TRAIN/VAL/TEST; seed 42')
    print('Soybean:', sb_manifest['totals'])

    for name, m in (('PlantDoc', pd_manifest), ('Soybean', sb_manifest)):
        print(f"{name}: classes={len(m['classes'])} "
              f"train/val/test = {m['totals']['train']}/{m['totals']['val']}/{m['totals']['test']}")


if __name__ == '__main__':
    main()
