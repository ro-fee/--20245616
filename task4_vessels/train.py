import argparse
from pathlib import Path
from PIL import Image
from protocol import training_fingerprint, check_warm_start
import copy
import json
import os
import random
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import (build_model, dice, eval_transform, list_ids, normalize, predict_prob,
                    preprocess, read_image, read_label)

torch.set_num_threads(4)
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/segmentation")
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="unet")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=-1)
    p.add_argument("--full", action="store_true", help="train one model on all 20 images")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--iters", type=int, default=40, help="batches per epoch")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--clahe", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--scale", type=float, default=1.0, help="upsample factor for train/infer")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay per step, 0 = off")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--init_run", default=None, help="Warm start from the same fold only")
    p.add_argument("--encoder_only_init", action="store_true")
    p.add_argument("--target512", action="store_true")
    p.add_argument("--encoder-weights", default=None)
    return p.parse_args()


class PatchDS(Dataset):
    def __init__(self, imgs, labels, n, crop):
        self.imgs, self.labels, self.n = imgs, labels, n
        self.aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.2, 0.3), rotate_limit=180,
                               border_mode=0, p=0.9),
            A.PadIfNeeded(crop, crop, border_mode=0),
            A.RandomCrop(crop, crop),
            A.HorizontalFlip(), A.VerticalFlip(),
            A.OneOf([A.RandomBrightnessContrast(0.25, 0.25), A.RandomGamma((70, 150)),
                     A.HueSaturationValue(10, 20, 15)], p=0.8),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 3)), A.GaussNoise(std_range=(0.01, 0.04))], p=0.25),
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        k = random.randrange(len(self.imgs))
        r = self.aug(image=self.imgs[k], mask=self.labels[k])
        return torch.from_numpy(normalize(r["image"])), torch.from_numpy(r["mask"][None].astype(np.float32))


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    d = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + d.mean()


def evaluate(model, imgs, labels, scale=1.0):
    """Mean Dice under the official 512 bilinear transform, threshold 0.5.
    imgs/labels are at original resolution; the model sees them upsampled by `scale`."""
    model.eval()
    probs = [predict_prob(model, im, scale) for im in imgs]
    s = np.mean([dice(eval_transform(p > 0.5), eval_transform(l)) for p, l in zip(probs, labels)])
    return s, probs


def rescale(img, lab, s):
    if s == 1.0:
        return img, lab
    h, w = lab.shape
    size = (round(w * s), round(h * s))
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    lab = (cv2.resize(lab.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    return img, lab


@torch.no_grad()
def ema_update(ema, model, d):
    for e, p in zip(ema.state_dict().values(), model.state_dict().values()):
        if e.dtype.is_floating_point:
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        else:
            e.copy_(p)


def train_one(args, tr_idx, va_idx, imgs, labels, tag):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(args.arch, args.encoder, pretrained=not bool(args.init_run), encoder_weights=args.encoder_weights).cuda()
    if args.init_run:
        init_cfg = json.load(open(os.path.join(args.init_run, 'args.json')))
        check_warm_start(vars(args), init_cfg)
        state=torch.load(os.path.join(args.init_run, tag+'.pt'), map_location='cuda', weights_only=True)
        if args.encoder_only_init:
            model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')})
        else:
            model.load_state_dict(state)
        print('Warm start same fold:', tag, 'train indices',tr_idx,'validation indices',va_idx,flush=True)
    sc = [rescale(imgs[i], labels[i], args.scale) for i in tr_idx]
    ds = PatchDS([a for a, _ in sc], [b for _, b in sc], args.iters * args.bs, args.crop)
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
                    pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs * args.iters
    warm = 3 * args.iters
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler()
    best, best_probs = -1, None
    path = os.path.join(args.out, f"{tag}.pt")
    for ep in range(args.epochs):
        model.train()
        t0, tl = time.time(), 0
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = loss_fn(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema_update(ema, model, args.ema)
            tl += loss.item()
        net = ema if ema is not None else model
        if va_idx and (ep + 1) % args.eval_every == 0:
            s, probs = evaluate(net, [imgs[i] for i in va_idx], [labels[i] for i in va_idx], args.scale)
            if s > best:
                best, best_probs = s, probs
                torch.save(net.state_dict(), path)
            print(f"[{tag}] ep {ep+1} loss {tl/args.iters:.4f} val_dice {s:.5f} best {best:.5f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        elif (ep + 1) % args.eval_every == 0:
            print(f"[{tag}] ep {ep+1} loss {tl/args.iters:.4f} ({time.time()-t0:.1f}s)", flush=True)
    if not va_idx:
        torch.save((ema if ema is not None else model).state_dict(), path)
    return best, best_probs


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    d = os.path.join(args.data, "train")
    ids = list_ids(os.path.join(d, "image"))
    if len(ids) != 20 or args.folds != 5:
        raise ValueError("This protocol requires 20 training images and five folds")
    if args.epochs < 4 or args.epochs % args.eval_every:
        raise ValueError("Epoch count must exceed warmup and end on an evaluation epoch")
    args.training_fingerprint = training_fingerprint(Path(d), ids)
    args.validation_ids = [[ids[j] for j in sorted(f.tolist())]
        for f in np.array_split(np.random.RandomState(0).permutation(len(ids)), args.folds)]
    record = vars(args)
    record_path = Path(args.out)/"args.json"
    if record_path.exists() and json.loads(record_path.read_text()) != record:
        raise ValueError("Output directory already belongs to another run")
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    raw = [read_image(os.path.join(d, "image", f"{i}.jpg")) for i in ids]
    if args.target512:
        raw = [np.array(Image.fromarray(im).resize((512,512),Image.Resampling.BILINEAR)) for im in raw]
    imgs = [preprocess(im,args.clahe) for im in raw]
    labels = [read_label(os.path.join(d, "label", f"{i}.jpg")) for i in ids]
    if args.target512:
        labels = [eval_transform(l).astype(np.uint8) for l in labels]
    if args.full:
        train_one(args, list(range(len(ids))), [], imgs, labels, "full")
        return
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ids))
    folds = np.array_split(perm, args.folds)
    json.dump(dict(split_seed=0,validation_ids=[[ids[j] for j in sorted(f.tolist())] for f in folds]),
              open(os.path.join(args.out,'split.json'),'w'),indent=2)
    scores = {}
    for k, va in enumerate(folds):
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        va = sorted(va.tolist())
        tr = [i for i in range(len(ids)) if i not in va]
        best, probs = train_one(args, tr, va, imgs, labels, f"fold{k}")
        np.savez_compressed(os.path.join(args.out, f"oof{k}.npz"), ids=np.array([ids[i] for i in va]),
                            probs=np.stack(probs))
        scores[k] = best
        print(f"fold {k} val {[ids[i] for i in va]} best dice {best:.5f}", flush=True)
    print("CV:", scores, "mean", np.mean(list(scores.values())), flush=True)


if __name__ == "__main__":
    main()
