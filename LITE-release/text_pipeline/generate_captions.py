"""Generate English image captions with a VLM (text stage of the text pipeline).

Stage 1/2 of the released text pipeline:
  1) this script  : image -> English caption (.txt) via a GLM-4V-family VLM;
  2) encode_text_features.py : caption (.txt) -> (1, L, 768) .npy feature via
     bert-base-uncased.

For every image under Image/{train,val,test}/<class>/ that lacks a caption, the
image is sent to the VLM with the prompt in prompt_caption_en.txt (same prompt
for every class; captions describe visible symptoms only, never the plant or
disease name), and the reply is saved to Text/<split>/<class>/<stem>.txt.

Safety / reproducibility:
  - the API key is read ONLY from the environment variable ZHIPU_API_KEY and is
    never written to any file;
  - the endpoint must be https, its host must be in the allow-list, and every
    resolved IP must be public (SSRF / DNS-rebinding protection);
  - redirects are disabled; 429 responses honor Retry-After;
  - existing .txt files are skipped, so the script is resumable;
  - every call is appended to a CSV log (--log).

Usage:
  export ZHIPU_API_KEY=...
  python text_pipeline/generate_captions.py --dataset-root <ds_root> --dry-run
  python text_pipeline/generate_captions.py --dataset-root <ds_root> \
      --model glm-4.6v
"""
import argparse
import base64
import csv
import io
import ipaddress
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

IMG_EXTS = {'.jpg', '.jpeg', '.png'}
PROMPT_PATH = Path(__file__).resolve().parent / 'prompt_caption_en.txt'
# Only the VLM endpoints actually used by the project (extend explicitly if needed)
ALLOWED_HOSTS = {'open.bigmodel.cn', 'dashscope.aliyuncs.com', 'api.openai.com'}

def safe_join(base, *parts):
    """Controlled path join: the resolved path must stay inside `base`."""
    base = Path(base).resolve()
    p = (base / Path(*parts)).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f'path escapes base: {parts}')
    return p


def validate_endpoint(base_url):
    """Require https + allow-listed host + public resolved IPs
    (SSRF / DNS-rebinding protection)."""
    u = urlparse(base_url)
    if u.scheme != 'https':
        raise ValueError(f'only https is allowed, got: {u.scheme}')
    host = u.hostname or ''
    if host not in ALLOWED_HOSTS:
        raise ValueError(f'host not in allow-list: {host}')
    port = u.port or 443
    for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP):
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f'endpoint resolved to a non-public address: {ip}')
    return f'https://{host}{u.path}'.rstrip('/')


def find_missing(root):
    """[(split, class, filename)] for images lacking a caption .txt."""
    out = []
    for split in ('train', 'val', 'test'):
        idir = root / 'Image' / split
        if not idir.is_dir():
            continue
        for cdir in sorted(p for p in idir.iterdir() if p.is_dir()):
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                txt = root / 'Text' / split / cdir.name / f'{f.stem}.txt'
                if not txt.exists():
                    out.append((split, cdir.name, f.name))
    return out

def call_vlm(api_key, endpoint, model, prompt, image_path, timeout=120, retries=3):
    """OpenAI-compatible chat endpoint; the image is sent as a base64 data URL.
    `endpoint` must have passed validate_endpoint()."""
    import requests
    path = Path(image_path)
    mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(
        path.suffix.lower(), 'image/jpeg')
    b64 = base64.b64encode(path.read_bytes()).decode()
    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url',
                 'image_url': {'url': f'data:{mime};base64,{b64}'}},
            ],
        }],
        'temperature': 0.3,   # sampling fixed as reported in the paper (Sec. 3.6)
        'max_tokens': 1024,
    }
    headers = {'Authorization': f'Bearer {api_key}',
               'Content-Type': 'application/json'}
    last = None
    for i in range(retries):
        try:
            r = requests.post(f'{endpoint}/chat/completions', headers=headers,
                              json=payload, timeout=timeout,
                              allow_redirects=False)
            if r.status_code == 429:
                # rate limit: honor Retry-After when present, else back off
                wait = int(r.headers.get('Retry-After', 5 * (i + 1) * (i + 1)))
                time.sleep(wait)
                last = RuntimeError(f'429: {r.text[:200]}')
                continue
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'].strip()
        except Exception as e:            # network / rate limit / parse errors
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f'VLM call failed after {retries} tries: {last}')

def main():
    ap = argparse.ArgumentParser(description='English caption generation (stage 1)')
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--model', default='glm-4.6v')
    ap.add_argument('--base-url', default='https://open.bigmodel.cn/api/paas/v4')
    ap.add_argument('--dry-run', action='store_true',
                    help='only list missing captions, no network access')
    ap.add_argument('--limit', type=int, default=0, help='process at most N images')
    ap.add_argument('--sleep', type=float, default=0.5,
                    help='seconds between API calls (rate limiting)')
    ap.add_argument('--classes', default='',
                    help='only process these classes (comma-separated)')
    ap.add_argument('--log', default='caption_generation_log.csv',
                    help='CSV log path (one row per API call)')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    missing = find_missing(root)
    if args.classes:
        keep = {c.strip() for c in args.classes.split(',') if c.strip()}
        missing = [m for m in missing if m[1] in keep]
    print(f'missing captions: {len(missing)}')
    if args.limit:
        missing = missing[:args.limit]

    if args.dry_run:
        for split, cls, name in missing[:50]:
            print(f'  {split:5s} {cls:30s} {name}')
        if len(missing) > 50:
            print(f'  ... ({len(missing) - 50} more)')
        return

    api_key = os.environ.get('ZHIPU_API_KEY')
    if not api_key:
        sys.exit('set the ZHIPU_API_KEY environment variable first '
                 '(never write the key into any file)')
    endpoint = validate_endpoint(args.base_url)
    prompt = PROMPT_PATH.read_text(encoding='utf-8')

    log_path = Path(args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for i, (split, cls, name) in enumerate(missing, 1):
        img = safe_join(root, 'Image', split, cls, name)
        txt = safe_join(root, 'Text', split, cls, f'{Path(name).stem}.txt')
        try:
            caption = call_vlm(api_key, endpoint, args.model, prompt, img)
            txt.parent.mkdir(parents=True, exist_ok=True)
            txt.write_text(caption, encoding='utf-8')
            log_rows.append({'image': f'{split}/{cls}/{name}', 'model': args.model,
                             'status': 'ok', 'chars': len(caption),
                             'timestamp': time.strftime('%F %T')})
            print(f'[{i}/{len(missing)}] ok {split}/{cls}/{name} '
                  f'({len(caption)} chars)')
        except Exception as e:
            log_rows.append({'image': f'{split}/{cls}/{name}', 'model': args.model,
                             'status': f'fail: {e}', 'chars': 0,
                             'timestamp': time.strftime('%F %T')})
            print(f'[{i}/{len(missing)}] FAIL {split}/{cls}/{name}: {e}')
        time.sleep(args.sleep)

    # append to the log (use a per-process --log file when running in parallel)
    prev = log_path.read_text(encoding='utf-8') if log_path.exists() else ''
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=['image', 'model', 'status', 'chars',
                                        'timestamp'])
    if not prev:
        w.writeheader()
    w.writerows(log_rows)
    log_path.write_text(prev + buf.getvalue(), encoding='utf-8')
    print('log ->', log_path)


if __name__ == '__main__':
    main()
