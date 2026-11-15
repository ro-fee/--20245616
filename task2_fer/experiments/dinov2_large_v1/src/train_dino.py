"""DINOv2-Large LoRA pilot. Only train_pixels.npy and the grouped train manifest are read.

Modes: selftest, prepare, smoke, extract, head, train.
Base weights are immutable; checkpoints contain only LoRA + classification head.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / 'experiments/dinov2_large_v1'
for key, val in {
    'TMPDIR': ROOT / 'tmp', 'XDG_CACHE_HOME': ROOT / 'cache',
    'HF_HOME': ROOT / 'cache/hf', 'TORCH_HOME': ROOT / 'cache/torch',
    'PIP_CACHE_DIR': ROOT / 'cache/pip', 'TRITON_CACHE_DIR': ROOT / 'cache/triton',
}.items():
    os.environ[key] = str(val)
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoModel, Dinov2Config, Dinov2Model
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

CLASS_MAP = {'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def emit(event, **data):
    row = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **data}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disk_check(extra=0):
    free = shutil.disk_usage(ROOT).free
    if free - extra < 15 * GIB:
        raise RuntimeError(f'Disk reserve would be below 15 GiB; free={free/GIB:.2f}, extra={extra/GIB:.2f}')


def gpu_setup(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 is required')
    free, total = torch.cuda.mem_get_info()
    if free < (args.memory_gib + 4) * GIB:
        raise RuntimeError(f'Insufficient free GPU memory for budget + reserve: {free/GIB:.2f} GiB')
    torch.cuda.set_per_process_memory_fraction(min(args.memory_gib * GIB / total, 0.95))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    emit('gpu_budget', free_gib=free/GIB, process_budget_gib=args.memory_gib)


def manifest(fold):
    df = pd.read_csv(ROOT / 'artifacts/train_manifest.csv')
    if not np.array_equal(df['index'].to_numpy(), np.arange(len(df))):
        raise ValueError('Manifest indices are not canonical pixel-array order')
    if not np.array_equal(df.emotion.map(CLASS_MAP).to_numpy(), df.label.to_numpy()):
        raise ValueError('Class mapping mismatch')
    tr, va = np.flatnonzero(df.fold.to_numpy() != fold), np.flatnonzero(df.fold.to_numpy() == fold)
    if set(df.iloc[tr].group) & set(df.iloc[va].group):
        raise ValueError('Duplicate group crosses validation fold')
    pixels = np.load(ROOT / 'data/train_pixels.npy', mmap_mode='r')
    if len(pixels) != len(df) or pixels.shape[1:] != (48, 48):
        raise ValueError('Training pixel cache mismatch')
    return df, tr, va


def metric(p, y):
    pred = p.argmax(1)
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=range(6), average='macro', zero_division=0)),
                recall=recall_score(y, pred, labels=range(6), average=None, zero_division=0).tolist())


class Faces(Dataset):
    def __init__(self, indices, labels, size, mean, std, augment=False):
        self.indices = np.asarray(indices)
        self.labels = labels
        self.pixels = np.load(ROOT / 'data/train_pixels.npy', mmap_mode='r')
        ops = [T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)]
        if augment:
            ops += [T.RandomHorizontalFlip(), T.RandomAffine(8, translate=(.04, .04), scale=(.95, 1.05),
                    interpolation=T.InterpolationMode.BILINEAR), T.ColorJitter(brightness=.12, contrast=.12)]
        ops += [T.ToTensor(), T.Normalize(mean, std)]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = Image.fromarray(self.pixels[idx]).convert('RGB')
        return self.transform(x), int(self.labels[idx]), idx


def worker_seed(_):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def loader(ds, args, shuffle=False, epoch=0):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=True, worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(args.seed + epoch), persistent_workers=False)


def provenance():
    p = EXP / 'artifacts/base_manifest.json'
    if not p.exists():
        raise RuntimeError('Run prepare to download the official base weights first')
    d = json.loads(p.read_text())
    if not d.get('download_complete'):
        raise RuntimeError('Base download is incomplete')
    return d


def processor_values(base):
    d = json.loads((Path(base) / 'preprocessor_config.json').read_text())
    return d['image_mean'], d['image_std']


def prepare(args):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    repo = args.repo
    if repo not in ('facebook/dinov2-large',):
        raise ValueError('Only the approved official candidates are supported')
    prepared = EXP / 'artifacts/base_manifest.json'
    if prepared.exists():
        existing = provenance()
        if existing['repo'] != repo:
            raise ValueError('Prepared repository differs')
        snapshot = Path(existing['snapshot']).resolve()
        if not snapshot.is_relative_to(ROOT.resolve()):
            raise ValueError('Base weights must stay inside the task workspace')
        for f in existing['weights'] + existing.get('config_files', []):
            if sha(snapshot / f['name']) != f['sha256']:
                raise ValueError('Prepared base checksum mismatch')
        emit('reuse_verified_base', repo=repo, revision=existing['revision'])
        return
    cache = str(ROOT / 'cache/hf/hub')
    pinned = EXP / 'artifacts/download_plan.json'
    revision = json.loads(pinned.read_text())['revision'] if pinned.exists() else '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c'
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
    write_json(pinned, dict(repo=repo, revision=info.sha))
    # This public repository is downloaded directly from the official source.
    hf_hub_download(repo, 'config.json', revision=info.sha, cache_dir=cache, token=False)
    files = [x for x in info.siblings if x.rfilename.endswith('.safetensors')]
    if not files or any(x.size is None for x in files):
        raise RuntimeError('Cannot establish exact weight size')
    size = sum(x.size for x in files)
    expected = []
    pending = 0
    for x in files:
        lfs_hash = x.lfs.sha256 if x.lfs else None
        cached = Path(cache) / ('models--' + repo.replace('/', '--')) / 'blobs' / str(lfs_hash)
        if not cached.exists() or cached.stat().st_size != x.size:
            pending += x.size
        expected.append({'name': x.rfilename, 'size': x.size, 'sha256': lfs_hash})
    disk_check(pending + 2 * GIB)
    emit('download_start', repo=repo, revision=info.sha, bytes=size, uncached_bytes=pending)
    path = snapshot_download(repo, revision=info.sha, cache_dir=cache, max_workers=1, token=False,
                             allow_patterns=['config.json', 'preprocessor_config.json', '*.safetensors', '*.safetensors.index.json'])
    for f in expected:
        got = sha(Path(path) / f['name'])
        if f['sha256'] and got != f['sha256']:
            raise RuntimeError('Downloaded base weight hash mismatch')
        f['sha256'] = got
    configs=[dict(name=p.name,sha256=sha(p)) for p in sorted(Path(path).glob('*.json'))]
    write_json(EXP / 'artifacts/base_manifest.json', dict(repo=repo, revision=info.sha, snapshot=path,
               config_files=configs,
               weights=expected, download_complete=True, pretraining_overlap='unknown',
               source='official public Hugging Face repository'))
    emit('download_verified', repo=repo, revision=info.sha)


class Classifier(nn.Module):
    def __init__(self, base, args, adapters=True):
        super().__init__()
        self.registers = getattr(base.config, 'num_register_tokens', 0)
        dim = base.config.hidden_size * 2
        base.requires_grad_(False)
        if adapters:
            names = [n for n, m in base.named_modules() if isinstance(m, nn.Linear) and n.endswith(('query', 'value'))]
            if len(names) != 2 * base.config.num_hidden_layers:
                raise ValueError('Unexpected query/value projection layout')
            base = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.alpha,
                     lora_dropout=args.lora_dropout, target_modules=['query', 'value'], bias='none'))
        self.backbone = base
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(.1), nn.Linear(dim, 6))

    def features(self, x):
        tokens = self.backbone(pixel_values=x).last_hidden_state
        return torch.cat((tokens[:, 0], tokens[:, 1+self.registers:].mean(1)), dim=-1).float()

    def forward(self, x):
        return self.head(self.features(x))


def load_model(args, adapters=True):
    p = provenance()
    base = AutoModel.from_pretrained(p['snapshot'], local_files_only=True, torch_dtype=torch.bfloat16,
                                    attn_implementation='sdpa', low_cpu_mem_usage=True)
    model = Classifier(base, args, adapters).cuda()
    if adapters:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    emit('model_loaded', total_parameters=sum(p.numel() for p in model.parameters()),
         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
         targets=['query', 'value'] if adapters else [], revision=p['revision'])
    return model


def identity(args):
    return dict(repo=provenance()['repo'], revision=provenance()['revision'], fold=args.fold,
        seed=args.seed, size=args.size, rank=args.rank, alpha=args.alpha, lora_dropout=args.lora_dropout,
        lr=args.lr, head_lr=args.head_lr, batch_size=args.batch_size, accumulation=args.accumulation,
        epochs=args.epochs, manifest_sha256=sha(ROOT / 'artifacts/train_manifest.csv'),
        source_sha256=sha(__file__))


def save_trainable(path, model, args, **extra):
    disk_check(.5 * GIB)
    data = dict(identity=identity(args), adapter=get_peft_model_state_dict(model.backbone),
                head=model.head.state_dict(), **extra)
    tmp = Path(str(path) + '.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def restore(path, model, args):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if d['identity'] != identity(args):
        raise ValueError('Checkpoint identity differs from current experiment')
    current = get_peft_model_state_dict(model.backbone)
    if set(d['adapter']) != set(current):
        raise ValueError('Adapter checkpoint keys mismatch')
    set_peft_model_state_dict(model.backbone, d['adapter'])
    model.head.load_state_dict(d['head'])
    return d


@torch.inference_mode()
def predict(model, dl, flip=False):
    model.eval()
    preds, indices = [], []
    for x, _, ix in dl:
        x = x.cuda(non_blocking=True).to(torch.bfloat16)
        if flip:
            x = x.flip(-1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(x).float().softmax(-1)
        preds.append(p.cpu().numpy())
        indices.append(ix.numpy())
    return np.concatenate(indices), np.concatenate(preds)


def compare(args, indices, probs):
    df, _, va = manifest(args.fold)
    if not np.array_equal(indices, va):
        raise ValueError('Validation predictions are out of order')
    return dict(fold=args.fold, single=metric(probs,df.label.to_numpy()[va]),
                validation_rows=len(va),basis='training_fold_only')


def extract(args):
    out = EXP / 'artifacts/train_features.npy'
    if out.exists() and out.with_suffix('.json').exists():
        meta = json.loads(out.with_suffix('.json').read_text())
        if (meta['revision'] == provenance()['revision'] and meta['size'] == args.size
                and meta['manifest_sha256'] == sha(ROOT / 'artifacts/train_manifest.csv')):
            emit('skip_features_complete'); return
        raise ValueError('Feature cache belongs to a different model or input size')
    df, _, _ = manifest(args.fold)
    model = load_model(args, adapters=False).eval()
    mean, std = processor_values(provenance()['snapshot'])
    dl = loader(Faces(np.arange(len(df)), df.label.to_numpy(), args.size, mean, std), args)
    dim = model.head[0].normalized_shape[0]
    partial = out.with_suffix('.partial.npy')
    features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16, shape=(len(df), dim))
    with torch.inference_mode():
        for step, (x, _, idx) in enumerate(dl):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f = model.features(x.cuda(non_blocking=True).to(torch.bfloat16))
            features[idx.numpy()] = f.cpu().numpy().astype(np.float16)
            if step % 100 == 0: emit('extract_progress', processed=int(idx[-1])+1, total=len(df))
    features.flush(); del features
    partial.replace(out)
    write_json(out.with_suffix('.json'), dict(revision=provenance()['revision'], size=args.size,
               shape=[len(df), dim], manifest_sha256=sha(ROOT / 'artifacts/train_manifest.csv')))
    emit('features_complete', path=str(out))


def train_head(args):
    df, tr, va = manifest(args.fold)
    feat = np.load(EXP / 'artifacts/train_features.npy', mmap_mode='r')
    meta = json.loads((EXP / 'artifacts/train_features.json').read_text())
    if (meta['revision'] != provenance()['revision'] or meta['size'] != args.size
            or meta['manifest_sha256'] != sha(ROOT / 'artifacts/train_manifest.csv')):
        raise ValueError('Stale features')
    head = nn.Sequential(nn.LayerNorm(feat.shape[1]), nn.Dropout(.1), nn.Linear(feat.shape[1], 6)).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01)
    y = df.label.to_numpy()
    best = (-1., -1.)
    saved = EXP / f'artifacts/head_fold{args.fold}.pt'
    if saved.exists():
        if torch.load(saved, map_location='cpu', weights_only=False)['identity'] != identity(args):
            raise ValueError('Completed head configuration differs')
        if not (EXP / f'artifacts/head_fold{args.fold}_comparison.json').exists():
            pred = np.load(EXP / f'artifacts/head_fold{args.fold}_preds.npz')
            write_json(EXP / f'artifacts/head_fold{args.fold}_comparison.json', compare(args, va, pred['val_orig']))
        emit('skip_head_complete'); return
    for epoch in range(30):
        head.train()
        order = np.random.default_rng(args.seed + epoch).permutation(tr)
        for ids in np.array_split(order, math.ceil(len(order)/256)):
            x = torch.from_numpy(np.array(feat[ids], dtype=np.float32)).cuda()
            target = torch.from_numpy(y[ids]).cuda()
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x), target, label_smoothing=.05)
            loss.backward(); opt.step()
        head.eval()
        with torch.inference_mode():
            p = np.concatenate([head(torch.from_numpy(np.array(feat[ids], dtype=np.float32)).cuda()).softmax(-1).cpu().numpy()
                    for ids in np.array_split(va, math.ceil(len(va)/256))])
        scores = metric(p, y[va]); key = (scores['accuracy'], scores['macro_f1'])
        emit('head_epoch', epoch=epoch+1, **scores)
        if key > best:
            best = key
            torch.save(dict(head=head.state_dict(), identity=identity(args), epoch=epoch+1), saved.with_suffix('.working.pt'))
            np.savez_compressed(EXP / f'artifacts/head_fold{args.fold}_preds.npz', valid_idx=va, val_orig=p)
    saved.with_suffix('.working.pt').replace(saved)
    pred = np.load(EXP / f'artifacts/head_fold{args.fold}_preds.npz')
    write_json(EXP / f'artifacts/head_fold{args.fold}_comparison.json', compare(args, va, pred['val_orig']))


def train(args, smoke=False):
    df, tr, va = manifest(args.fold)
    y = df.label.to_numpy()
    run = EXP / ('artifacts/smoke' if smoke else f'artifacts/lora_fold{args.fold}')
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'complete.json').exists() and not smoke:
        if json.loads((run / 'complete.json').read_text())['identity'] != identity(args):
            raise ValueError('Completed fold configuration differs')
        emit('skip_training_complete'); return
    model = load_model(args)
    write_json(run / 'config.json', identity(args))
    head_path = EXP / f'artifacts/head_fold{args.fold}.pt'
    if head_path.exists() and not smoke:
        d = torch.load(head_path, map_location='cpu', weights_only=False)
        if d['identity'] != identity(args): raise ValueError('Head identity mismatch')
        model.head.load_state_dict(d['head'])
    opt = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': args.lr, 'weight_decay': .01},
        {'params': list(model.head.parameters()), 'lr': args.head_lr, 'weight_decay': .01}])
    mean, std = processor_values(provenance()['snapshot'])
    train_ds = Faces(tr, y, args.size, mean, std, True)
    valid_dl = loader(Faces(va, y, args.size, mean, std), args)
    total_updates = args.epochs * math.ceil(math.ceil(len(tr)/args.batch_size)/args.accumulation)
    warmup = max(1, int(.05 * total_updates))
    def factor(step):
        if step < warmup: return max(.05, (step+1)/warmup)
        return .05 + .95 * (1+math.cos(math.pi*min(1., (step-warmup)/max(1,total_updates-warmup))))/2
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    epoch0, best, elapsed = 0, (-1., -1.), 0.
    if (run/'last.pt').exists() and not smoke:
        d = restore(run/'last.pt', model, args)
        opt.load_state_dict(d['optimizer']); schedule.load_state_dict(d['schedule'])
        epoch0, best, elapsed = d['epoch'], tuple(d['best']), d['elapsed']
        emit('resume', epoch_completed=epoch0)
    started = time.monotonic()
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epoch0, args.epochs):
        seed_all(args.seed + epoch)
        dl = loader(train_ds, args, True, epoch)
        model.train(); opt.zero_grad(set_to_none=True)
        loss_sum, seen, micro = 0., 0, 0
        max_steps = min(len(dl), args.smoke_steps) if smoke else len(dl)
        for step, (x, target, _) in enumerate(itertools.islice(dl, max_steps)):
            x, target = x.cuda(non_blocking=True).to(torch.bfloat16), target.cuda(non_blocking=True)
            window = min(args.accumulation, max_steps - (step // args.accumulation) * args.accumulation)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model(x), target, label_smoothing=.05)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            (loss/window).backward()
            if step == 0:
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                grads = [p.grad for n,p in model.named_parameters() if 'lora_B' in n and p.grad is not None]
                if not grads or not any(bool(g.abs().sum()>0) for g in grads):
                    raise RuntimeError('No nonzero LoRA gradient; checkpointing/configuration is broken')
            if (step+1) % args.accumulation == 0 or step+1 == max_steps:
                nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                opt.step(); schedule.step(); opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach()) * len(target); seen += len(target); micro += 1
            if step % 50 == 0 or step+1 == max_steps:
                emit('train_progress', smoke=smoke, fold=args.fold, epoch=epoch+1, step=step+1,
                     steps=max_steps, loss=loss_sum/seen, peak_gib=torch.cuda.max_memory_reserved()/GIB)
        if smoke:
            model.eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): expected=model(x).float().cpu()
            save_trainable(run/'roundtrip.pt', model, args)
            with torch.no_grad(): model.head[-1].bias.add_(1.)
            restore(run/'roundtrip.pt', model, args)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): actual=model(x).float().cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            result = dict(steps=micro, images=seen, seconds=time.monotonic()-started,
                          peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB, reload_matches=True,
                          lora_gradient_checked=True, identity=identity(args))
            write_json(EXP/'artifacts/smoke_pass.json', result); emit('smoke_pass', **result); return
        disk_check(.5 * GIB)
        ix, p = predict(model, valid_dl)
        assert np.array_equal(ix, va)
        scores = metric(p, y[va]); key = scores['accuracy'], scores['macro_f1']
        if key > best:
            best = key
            save_trainable(run/'best.pt', model, args, epoch=epoch+1, metrics=scores)
        elapsed_now = elapsed + time.monotonic()-started
        save_trainable(run/'last.pt', model, args, epoch=epoch+1, best=best, elapsed=elapsed_now,
                       optimizer=opt.state_dict(), schedule=schedule.state_dict())
        rec = emit('lora_epoch', fold=args.fold, epoch=epoch+1, train_loss=loss_sum/seen,
                   val=scores, elapsed_seconds=elapsed_now)
        with (run/'epochs.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
        if elapsed_now > args.max_hours*3600: raise RuntimeError('Experiment runtime budget reached; checkpoint saved')
    restore(run/'best.pt', model, args)
    ix, plain = predict(model, valid_dl); ix2, flip = predict(model, valid_dl, True)
    assert np.array_equal(ix, ix2) and np.array_equal(ix, va)
    np.savez_compressed(run/'preds.npz', valid_idx=ix, val_orig=plain, val_flip=flip)
    report = dict(plain=compare(args, ix, plain), flip_average=compare(args, ix, (plain+flip)/2))
    write_json(run/'comparison.json', report)
    write_json(run/'complete.json', dict(identity=identity(args), best=best, checkpoint_sha256=sha(run/'best.pt')))
    emit('fold_complete', **report)


def selftest(args):
    torch.set_num_threads(2)
    c = Dinov2Config(hidden_size=64, mlp_ratio=2, num_hidden_layers=2,
                      num_attention_heads=4, image_size=28, patch_size=14)
    base = Dinov2Model(c)
    frozen = {n: p.detach().clone() for n,p in base.named_parameters()}
    model = Classifier(base, args)
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train(); x=torch.randn(2,3,28,28); target=torch.tensor([1,4])
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    loss=nn.functional.cross_entropy(model(x),target); loss.backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
    opt.step()
    for n,p in model.backbone.get_base_model().named_parameters():
        if 'lora_' in n: continue
        key=n.replace('.base_layer.', '.')
        torch.testing.assert_close(p, frozen[key],rtol=0,atol=0)
    state={k:v.detach().clone() for k,v in get_peft_model_state_dict(model.backbone).items()}
    model.eval()
    with torch.inference_mode(): before=model(x)
    base2=Dinov2Model(c); base2.load_state_dict(frozen)
    other=Classifier(base2,args); set_peft_model_state_dict(other.backbone,state)
    other.head.load_state_dict(model.head.state_dict()); other.eval()
    with torch.inference_mode(): after=other(x)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    df,tr,va=manifest(args.fold)
    result=emit('selftest_pass', frozen_unchanged=True, lora_receives_gradients=True,
                reconstructed_predictions_match=True, train_rows=len(tr), validation_rows=len(va),
                group_overlap=0, total_params=sum(p.numel() for p in model.parameters()))
    write_json(EXP/'artifacts/selftest.json',result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','selftest','smoke','extract','head','train'])
    p.add_argument('--repo',default='facebook/dinov2-large')
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--size',type=int,default=224)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--rank',type=int,default=16)
    p.add_argument('--alpha',type=int,default=32)
    p.add_argument('--lora-dropout',type=float,default=.05)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--smoke-steps',type=int,default=100)
    p.add_argument('--memory-gib',type=float,default=24)
    p.add_argument('--max-hours',type=float,default=24)
    args=p.parse_args()
    if not 0<=args.fold<5 or args.size%14 or args.accumulation<1: raise ValueError('Invalid configuration')
    (EXP/'artifacts').mkdir(parents=True,exist_ok=True)
    if args.mode not in ('prepare', 'selftest'):
        if provenance()['repo'] != args.repo:
            raise ValueError('Prepared model does not match requested repo')
    seed_all(args.seed); disk_check()
    if args.mode=='prepare': return prepare(args)
    if args.mode=='selftest': return selftest(args)
    gpu_setup(args)
    if args.mode=='smoke': return train(args,smoke=True)
    if args.mode=='extract': return extract(args)
    if args.mode=='head': return train_head(args)
    if args.mode=='train':
        if not (EXP/'artifacts/smoke_pass.json').exists(): raise RuntimeError('Real-model smoke test required')
        smoke=json.loads((EXP/'artifacts/smoke_pass.json').read_text())
        if smoke['identity']!=identity(args): raise RuntimeError('Smoke result configuration differs')
        return train(args)


if __name__=='__main__':
    main()
