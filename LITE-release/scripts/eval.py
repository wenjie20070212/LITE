"""Evaluate a checkpoint on the TEST split exactly once.

Rebuilds the model from --task, loads best.pth (selected on validation),
evaluates on Image/test, and writes <out>/metrics.json plus
<out>/predictions.csv (per-sample correctness, ready for paired bootstrap).

Examples:
  python scripts/eval.py --ckpt checkpoints/soybean/mlp_matched_seed7/best.pth \
      --task student --dataset-path <soybean_root> --num-classes 8 --out eval_out
  python scripts/eval.py --ckpt runs/soybean_teacher/best.pth --task teacher \
      --dataset-path <soybean_root> --num-classes 8 --out eval_teacher
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.distillation import ResTextStudent, ResTextTeacher
from utils.img_process import build_loaders, eval_image_model, eval_teacher_model


def _guarded_out(path):
    p = Path(path)
    if '..' in p.parts:
        raise ValueError(f'unsafe output path: {path}')
    return p


def main():
    p = argparse.ArgumentParser(description='LITE test-set evaluation')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--task', required=True, choices=['teacher', 'student'])
    p.add_argument('--dataset-path', required=True)
    p.add_argument('--num-classes', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--name', default=None, help='run name recorded in metrics.json')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    _guarded_out(args.out)
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}

    if args.task == 'teacher':
        model = ResTextTeacher(num_classes=args.num_classes, model_size='x2_0').to(device)
        model.load_state_dict(sd, strict=True)
        eval_fn = eval_teacher_model
    else:
        model = ResTextStudent(num_classes=args.num_classes).to(device)
        model.load_state_dict(sd, strict=True)
        eval_fn = eval_image_model
    model.eval()

    loader = build_loaders(args.dataset_path, modes=('test',), num_workers=2)['test']
    pred_path = os.path.join(args.out, 'predictions.csv')
    acc = eval_fn(model, loader, device, save_to=pred_path)

    report = {'name': args.name or args.ckpt, 'task': args.task,
              'ckpt': args.ckpt, 'n_test': len(loader.dataset),
              'test_acc': round(acc, 4),
              'best_val_acc': ckpt.get('best_acc'), 'best_epoch': ckpt.get('epoch')}
    (Path(args.out) / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(f"[{report['name']}] TEST acc={acc:.2f}% on {len(loader.dataset)} images "
          f"(best_val={ckpt.get('best_acc')}% @epoch {ckpt.get('epoch')})")


if __name__ == '__main__':
    main()
