"""Encode English captions into BERT token-level features (stage 2/2).

For every caption Text/{train,val,test}/<class>/<stem>.txt that lacks a cached
feature, encodes the text with bert-base-uncased and stores a float32 .npy of
shape (1, 128, 768) under Textfeature/{split}/{class}/<stem>.npy.

Encoding spec (fixed; matches what the released models consume):
  - tokenizer : bert-base-uncased (google-bert/bert-base-uncased)
  - max_length=128, padding="max_length", truncation=True  (longer captions
    are hard-truncated at 128 tokens)
  - the model's last hidden states are kept per token (no pooling here);
    every token vector (including [PAD] positions) is L2-normalized
  - the training code averages token vectors WITHOUT an attention mask, i.e.
    [PAD] positions participate in the mean; this mirrors the original
    implementation and is documented here for transparency
  - output dtype float32, shape (1, 128, 768) (leading batch dim kept for
    dataloader compatibility)

Usage:
  python text_pipeline/encode_text_features.py --dataset-root <ds_root> \
      --bert-dir /path/to/bert-base-uncased
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png'}
MAX_LENGTH = 128

def safe_join(base, *parts):
    """Controlled path join: the resolved path must stay inside `base`."""
    base = Path(base).resolve()
    p = (base / Path(*parts)).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f'path escapes base: {parts}')
    return p


def find_missing(root):
    """[(split, class, stem)] for captions lacking a cached .npy feature."""
    out = []
    for split in ('train', 'val', 'test'):
        tdir = root / 'Text' / split
        if not tdir.is_dir():
            continue
        for cdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix != '.txt':
                    continue
                npy = root / 'Textfeature' / split / cdir.name / f'{f.stem}.npy'
                if not npy.exists():
                    out.append((split, cdir.name, f.stem))
    return out


def encode_texts(bert_dir, texts, max_length=MAX_LENGTH):
    """(N, max_length, 768) float32: per-token last hidden states, each
    L2-normalized over the hidden dimension (PAD positions included)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(bert_dir), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(bert_dir), local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mdl.to(device).eval()
    inputs = tok(list(texts), max_length=max_length, padding='max_length',
                 truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        hidden = mdl(**inputs).last_hidden_state
    hidden = F.normalize(hidden, p=2, dim=-1)
    return hidden.detach().cpu().numpy().astype('float32')

def main():
    ap = argparse.ArgumentParser(description='Caption -> .npy encoding (stage 2)')
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--bert-dir', required=True,
                    help='local directory of bert-base-uncased '
                         '(config.json + vocab.txt + model weights)')
    ap.add_argument('--limit', type=int, default=0, help='encode at most N items')
    ap.add_argument('--batch', type=int, default=8, help='encoding batch size')
    args = ap.parse_args()

    if not os.path.isdir(args.bert_dir):
        sys.exit(f'bert directory not found: {args.bert_dir} '
                 '(see README "Text feature pipeline" for the download command)')

    root = Path(args.dataset_root)
    todo = find_missing(root)
    print(f'captions to encode: {len(todo)}')
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print('nothing to do')
        return

    texts = [safe_join(root, 'Text', s, c, f'{stem}.txt').read_text(
        encoding='utf-8') for s, c, stem in todo]
    for k in range(0, len(todo), args.batch):
        chunk = todo[k:k + args.batch]
        sub = encode_texts(args.bert_dir, texts[k:k + args.batch])
        for j, (split, cls, stem) in enumerate(chunk):
            dst = safe_join(root, 'Textfeature', split, cls, f'{stem}.npy')
            dst.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            np.save(buf, sub[j][None, ...])
            dst.write_bytes(buf.getvalue())
        print(f'encoded {k + len(chunk)}/{len(todo)}')
    print('done')


if __name__ == '__main__':
    main()
