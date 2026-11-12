"""Dataset loading, augmentation, seeding, and evaluation helpers (release).

Dataset layout expected on disk (per dataset):

    <root>/Image/{train,val,test}/<class_name>/<image>.jpg
    <root>/Text/{train,val,test}/<class_name>/<stem>.txt          (raw captions)
    <root>/Textfeature/{train,val,test}/<class_name>/<stem>.npy   (1, L, 768)

Each sample is an (image, cached_text_feature, label) triple; images without a
matching .npy feature are skipped. Class order is fixed to sorted() so label
indices are identical across train/val/test and across runs.
"""
import os
import random
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def _safe_child(base, *parts):
    """Path containment guard: the resolved path must stay inside `base`."""
    base = Path(base).resolve()
    p = (base.joinpath(*parts)).resolve()
    if not p.is_relative_to(base):
        raise ValueError(f'path escapes dataset root: {parts}')
    return p


def _safe_out(path):
    """Guard for caller-provided output paths: reject '..' components."""
    p = Path(path)
    if '..' in p.parts:
        raise ValueError(f'unsafe output path: {path}')
    return p


class SortedPairDataset(Dataset):
    """(image, text_feature, label) triples with a deterministic class order.

    The class list is sorted(), so the label map is identical for every split
    built from the same dataset. Samples whose .npy feature is missing are
    skipped at load time.
    """

    def __init__(self, dataset_path, mode='train', transform=None,
                 textfeature_folder='Textfeature'):
        self.dataset_path = dataset_path
        self.mode = mode
        self.transform = transform
        self.image_dir = _safe_child(dataset_path, 'Image', mode)
        self.feature_dir = _safe_child(dataset_path, textfeature_folder, mode)
        self.classes = sorted(d for d in os.listdir(self.image_dir)
                              if os.path.isdir(os.path.join(self.image_dir, d)))
        self.image_paths, self.feature_paths, self.labels = [], [], []
        for idx, class_name in enumerate(self.classes):
            ci = _safe_child(self.image_dir, class_name)
            cf = _safe_child(self.feature_dir, class_name)
            for img_name in sorted(os.listdir(ci)):
                if os.path.splitext(img_name)[1].lower() not in IMG_EXTS:
                    continue
                feat = _safe_child(cf, os.path.splitext(img_name)[0] + '.npy')
                if feat.exists():
                    self.image_paths.append(str(ci / img_name))
                    self.feature_paths.append(str(feat))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        feature = torch.tensor(np.load(self.feature_paths[idx]), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, feature, self.labels[idx]


def build_transforms():
    """Augmentation: train = resize 256 + random resized crop 224 + h-flip;
    val/test = resize 256 + center crop 224; ImageNet normalization."""
    train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train, test


def build_loaders(dataset_path, batch_size=64, modes=('train', 'val', 'test'),
                  num_workers=4):
    t_train, t_test = build_transforms()
    loaders = {}
    for m in modes:
        ds = SortedPairDataset(
            dataset_path, mode=m, transform=(t_train if m == 'train' else t_test))
        loaders[m] = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(m == 'train'),
            num_workers=num_workers, drop_last=(m == 'train'), pin_memory=True)
    return loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_predictions_csv(save_to, rows):
    """Render predictions to CSV in memory, then write via a guarded path."""
    _safe_out(save_to)
    buf = StringIO()
    buf.write('true,pred,correct\n')
    for r in rows:
        buf.write(','.join(map(str, r)) + '\n')
    Path(save_to).write_text(buf.getvalue())


@torch.no_grad()
def eval_image_model(model, loader, device, save_to=None):
    """Evaluate an image-only model (student). Optionally dump per-sample
    predictions (true,pred,correct) as CSV."""
    model.eval()
    correct, total = 0, 0
    rows = []
    for images, _feat, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        _, pred = logits.max(1)
        ok = pred.eq(labels)
        correct += ok.sum().item()
        total += labels.size(0)
        if save_to is not None:
            for j in range(labels.size(0)):
                rows.append((int(labels[j].item()), int(pred[j].item()),
                             int(ok[j].item())))
    if save_to is not None:
        _write_predictions_csv(save_to, rows)
    return 100. * correct / max(total, 1)


@torch.no_grad()
def eval_teacher_model(model, loader, device, save_to=None):
    """Evaluate the multimodal teacher (text input consumed)."""
    model.eval()
    correct, total = 0, 0
    rows = []
    for images, text, labels in loader:
        images, labels = images.to(device), labels.to(device)
        text = text.squeeze(1).to(device)
        logits = model(text, images)
        _, pred = logits.max(1)
        ok = pred.eq(labels)
        correct += ok.sum().item()
        total += labels.size(0)
        if save_to is not None:
            for j in range(labels.size(0)):
                rows.append((int(labels[j].item()), int(pred[j].item()),
                             int(ok[j].item())))
    if save_to is not None:
        _write_predictions_csv(save_to, rows)
    return 100. * correct / max(total, 1)
