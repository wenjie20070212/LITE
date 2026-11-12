"""Train the LITE teacher and the main (width-matched) student model.

Protocol: Adam(lr 1e-3, wd 1e-4), batch 64, 200 epochs,
StepLR(step 50, gamma 0.5), grad clip 1.0. Checkpoints are selected by the
highest VALIDATION accuracy; the TEST split is only touched by scripts/eval.py.
"""
import argparse
import csv
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.distillation import MultiLevelDistillationLoss, ResTextStudent, ResTextTeacher
from utils.img_process import build_loaders, eval_image_model, eval_teacher_model, set_seed

def _guarded_out(path):
    """Reject '..' components in caller-provided output paths."""
    p = Path(path)
    if '..' in p.parts:
        raise ValueError(f'unsafe output path: {path}')
    return p


def _write_history_csv(out_dir, history):
    """Render history rows to CSV in memory, then write via a guarded path."""
    _guarded_out(out_dir)
    buf = StringIO()
    w = csv.DictWriter(buf, fieldnames=list(history[-1].keys()))
    w.writeheader()
    w.writerows(history)
    (Path(out_dir) / 'train_history.csv').write_text(buf.getvalue())


def _write_run_summary(out_dir, payload):
    _guarded_out(out_dir)
    (Path(out_dir) / 'run_summary.json').write_text(
        json.dumps(payload, indent=2, default=str))

def train_teacher(args, device, loaders):
    set_seed(args.seed)
    model = ResTextTeacher(num_classes=args.num_classes, model_size='x2_0').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    history, best_acc = [], 0.0
    for epoch in range(args.epochs):
        model.train()
        correct, total, running = 0, 0, 0.0
        for images, text, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            text = text.squeeze(1).to(device)
            logits = model(text, images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += logits.max(1)[1].eq(labels).sum().item()
        scheduler.step()
        val_acc = eval_teacher_model(model, loaders['val'], device)
        history.append({'epoch': epoch, 'train_acc': 100. * correct / max(total, 1),
                        'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']})
        print(f"epoch {epoch} train {history[-1]['train_acc']:.2f}% "
              f"val {val_acc:.2f}% loss {running / max(total, 1):.4f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'best_acc': best_acc,
                        'epoch': epoch, 'args': vars(args)}, args.ckpt_path)
    return best_acc, history

def build_teacher(args, device):
    """Load the multimodal teacher checkpoint (module. prefix stripped)."""
    ckpt = torch.load(args.teacher_checkpoint, map_location='cpu',
                      weights_only=True)
    sd = ckpt.get('state_dict', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model = ResTextTeacher(num_classes=args.num_classes, model_size='x2_0').to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"teacher loaded: {args.teacher_checkpoint} "
          f"(best_val_acc={ckpt.get('best_acc', 'NA')})")
    return model

def build_student_criterion(args):
    """Main-model loss. --lambda-text 0 = lambda_T ablation: the text-alignment
    term is dropped (weight 0) and the feature denominator is fixed at 1.0 so
    the remaining terms keep exactly their base scale (no re-normalization)."""
    if args.lambda_text > 0:
        return MultiLevelDistillationLoss(
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            temperature=args.temperature)
    return MultiLevelDistillationLoss(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        temperature=args.temperature,
        feature_weights={'img_features': 0.3, 'text_features': 0.0,
                         'fused_features': 0.5},
        feature_denominator=1.0)

def train_student(args, device, loaders, teacher):
    set_seed(args.seed)
    student = ResTextStudent(num_classes=args.num_classes).to(device)
    criterion = build_student_criterion(args)
    optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    history, best_acc = [], 0.0
    for epoch in range(args.epochs):
        student.train()
        correct, total, running = 0, 0, 0.0
        for images, text, labels in loaders['train']:
            images, labels = images.to(device), labels.to(device)
            text = text.squeeze(1).to(device)
            with torch.no_grad():
                t_logits, t_feat = teacher(text, images, return_features=True)
            s_logits, s_feat = student(images, return_features=True)
            loss, _ = criterion(s_logits, t_logits, labels, s_feat, t_feat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += s_logits.max(1)[1].eq(labels).sum().item()
        scheduler.step()
        val_acc = eval_image_model(student, loaders['val'], device)
        history.append({'epoch': epoch, 'train_acc': 100. * correct / max(total, 1),
                        'val_acc': val_acc, 'lr': optimizer.param_groups[0]['lr']})
        print(f"epoch {epoch} train {history[-1]['train_acc']:.2f}% "
              f"val {val_acc:.2f}% loss {running / max(total, 1):.4f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': student.state_dict(), 'best_acc': best_acc,
                        'epoch': epoch, 'history': history, 'args': vars(args)},
                       args.ckpt_path)
    return best_acc, history

def main():
    p = argparse.ArgumentParser(description='LITE training (release scripts)')
    p.add_argument('--task', required=True, choices=['teacher', 'student'])
    p.add_argument('--dataset-path', required=True,
                   help='dataset root containing Image/ and Textfeature/')
    p.add_argument('--num-classes', type=int, required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--lr-step-size', type=int, default=50)
    p.add_argument('--lr-gamma', type=float, default=0.5)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--temperature', type=float, default=8.0)
    p.add_argument('--lambda-text', type=float, default=0.2,
                   help='weight of the text-alignment term (lambda_T); '
                        '0 runs the lambda_T=0 ablation')
    p.add_argument('--device', default='cuda')
    p.add_argument('--teacher-checkpoint', default=None,
                   help='required for --task student')
    args = p.parse_args()

    if args.task == 'student' and not args.teacher_checkpoint:
        p.error('--teacher-checkpoint is required for --task student')
    if args.task == 'teacher' and args.teacher_checkpoint:
        p.error('--teacher-checkpoint is only valid for --task student')
    if args.task == 'student' and args.lambda_text < 0:
        p.error('--lambda-text must be >= 0')

    device = args.device if torch.cuda.is_available() else 'cpu'
    _guarded_out(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    args.ckpt_path = os.path.join(args.out_dir, 'best.pth')
    print(f"task={args.task} dataset={args.dataset_path} seed={args.seed} "
          f"protocol=Adam({args.lr},{args.wd}) bs{args.batch_size} {args.epochs}ep "
          f"StepLR({args.lr_step_size}x{args.lr_gamma})", flush=True)

    loaders = build_loaders(args.dataset_path, batch_size=args.batch_size,
                            modes=('train', 'val'))
    for m, l in loaders.items():
        print(f"  {m}: {len(l.dataset)} samples", flush=True)

    t0 = time.time()
    if args.task == 'teacher':
        best_acc, history = train_teacher(args, device, loaders)
    else:
        teacher = build_teacher(args, device)
        best_acc, history = train_student(args, device, loaders, teacher)
    _write_history_csv(args.out_dir, history)
    _write_run_summary(args.out_dir, {
        'best_val_acc': best_acc, 'seed': args.seed, 'task': args.task,
        'lambda_text': args.lambda_text, 'elapsed_sec': time.time() - t0,
        'args': vars(args)})
    print(f"DONE best_val_acc={best_acc:.2f}% elapsed={time.time() - t0:.0f}s",
          flush=True)


if __name__ == '__main__':
    main()
