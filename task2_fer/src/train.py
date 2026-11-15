from __future__ import annotations
import hashlib
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import (
    EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights,
    resnet18, efficientnet_b0, convnext_tiny,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
NUM_CLASSES = 6
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 160
SEED = 20261011

def repeat_channels(x):
    return x.repeat(3,1,1)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

def make_transform(train):
    steps = [T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BICUBIC)]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=8, translate=(0.04, 0.04),
                           scale=(0.95, 1.05), interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.12, contrast=0.12),
        ]
    steps += [T.ToTensor(), T.Lambda(repeat_channels),
              T.Normalize(MEAN, STD)]
    if train:
        steps += [T.RandomErasing(p=0.08, scale=(0.01, 0.03))]
    return T.Compose(steps)

class FaceDataset(Dataset):
    def __init__(self, path, indices, labels, train):
        self.pixels = np.load(path, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.transform = make_transform(train)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, position):
        index = int(self.indices[position])
        image = Image.fromarray(self.pixels[index], mode="L")
        target = -1 if self.labels is None else int(self.labels[index])
        return self.transform(image), target, index

def make_model(name):
    if name == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    elif name == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
    else:
        raise ValueError(name)
    return model.cuda().to(memory_format=torch.channels_last)

def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))

def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True, persistent_workers=True,
                      worker_init_fn=seed_worker,
                      generator=generator)

@torch.inference_mode()
def predict(model, loader, flipped=False):
    model.eval()
    probs = []
    indices = []
    for x, _, index in loader:
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
        if flipped:
            x = x.flip(-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        probs.append(logits.float().softmax(dim=1).cpu().numpy())
        indices.append(index.numpy())
    return np.concatenate(indices), np.concatenate(probs)

def metrics(probs, labels):
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "recall": [float(x) for x in recall_score(
            labels, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)]
    }

def train_fold(args, fold, labels, folds):
    pred_file = ART / f"{args.model}_fold{fold}_preds.npz"
    ckpt_file = ART / f"{args.model}_fold{fold}.pt"
    receipt = ART / f"{args.model}_fold{fold}_complete.json"
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    identity=dict(args=vars(args),fold=fold,manifest=digest(ART/'train_manifest.csv'),
                  pixels=digest(ROOT/'data/train_pixels.npy'),code=digest(Path(__file__)))
    if pred_file.exists() or ckpt_file.exists() or receipt.exists():
        if not (receipt.exists() and pred_file.exists() and ckpt_file.exists()):
            raise ValueError("Incomplete run: choose a fresh artifacts directory")
        record=json.loads(receipt.read_text())
        if record['identity']!=identity or record['checkpoint']!=digest(ckpt_file) or record['predictions']!=digest(pred_file):
            raise ValueError("Cached run identity or contents changed")
        print("skip_verified", args.model, fold, flush=True)
        return
    seed = SEED + 100 * fold + {"resnet18": 1, "efficientnet_b0": 2, "convnext_tiny": 3}[args.model]
    seed_everything(seed)
    train_idx = np.flatnonzero(folds != fold)
    valid_idx = np.flatnonzero(folds == fold)
    train_set = FaceDataset(ROOT / "data" / "train_pixels.npy", train_idx, labels, True)
    valid_set = FaceDataset(ROOT / "data" / "train_pixels.npy", valid_idx, labels, False)
    train_loader = make_loader(train_set, args.batch_size, True, seed)
    valid_loader = make_loader(valid_set, args.batch_size, False, seed)

    model = make_model(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))
    def multiplier(step):
        if step < warmup:
            return max(0.05, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * min(1.0, progress))) / 2
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    log_file = ROOT / "logs" / f"{args.model}_fold{fold}.jsonl"
    best_acc = -1.0
    best_f1 = -1.0
    patience = 0
    for epoch in range(args.epochs):
        started = time.monotonic()
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, y, _ in train_loader:
            x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            schedule.step()
            loss_sum += float(loss.item()) * len(y)
            seen += len(y)
        order, val_probs = predict(model, valid_loader)
        assert np.array_equal(order, valid_idx)
        val_metrics = metrics(val_probs, labels[valid_idx])
        record = {
            "model": args.model, "fold": fold, "epoch": epoch + 1,
            "train_loss": loss_sum / seen, "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.monotonic() - started, 2),
        }
        with log_file.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        acc, f1 = val_metrics["accuracy"], val_metrics["macro_f1"]
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc, best_f1 = acc, f1
            patience = 0
            tmp = ckpt_file.with_suffix(".tmp")
            torch.save({
                "state_dict": model.state_dict(), "model": args.model,
                "fold": fold, "epoch": epoch + 1, "metrics": val_metrics,
                "config": vars(args), "seed": seed,
            }, tmp)
            os.replace(tmp, ckpt_file)
        else:
            patience += 1
            if patience >= args.patience:
                break
    saved = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    valid_order, val_orig = predict(model, valid_loader)
    flip_order, val_flip = predict(model, valid_loader, flipped=True)
    assert np.array_equal(valid_order, flip_order) and np.array_equal(valid_order, valid_idx)
    np.savez_compressed(pred_file, valid_idx=valid_order, val_orig=val_orig,
                        val_flip=val_flip)
    receipt.write_text(json.dumps(dict(identity=identity,checkpoint=digest(ckpt_file),predictions=digest(pred_file)),indent=2))
    print("fold_done", args.model, fold, saved["metrics"], flush=True)
    del model
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "efficientnet_b0", "convnext_tiny"], required=True)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    (ROOT/"logs").mkdir(parents=True,exist_ok=True)
    with (ART / "train_manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    folds = np.array([int(row["fold"]) for row in rows], dtype=np.int64)
    selected = range(5) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    (ART / f"{args.model}_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    for fold in selected:
        assert fold in range(5)
        train_fold(args, fold, labels, folds)

if __name__ == "__main__":
    main()
