import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key, folder in {'HF_HOME':'cache/hf', 'HF_HUB_CACHE':'cache/hf/hub',
                    'TORCH_HOME':'cache/torch', 'XDG_CACHE_HOME':'cache',
                    'MPLCONFIGDIR':'cache/matplotlib', 'TMPDIR':'tmp',
                    'CUDA_CACHE_PATH':'cache/cuda', 'TORCH_EXTENSIONS_DIR':'cache/extensions',
                    'TRITON_CACHE_DIR':'cache/triton', 'TORCHINDUCTOR_CACHE_DIR':'cache/inductor'}.items():
    p = ROOT / folder
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

CLASSES = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
torch.hub.set_dir(str(ROOT / 'cache/torch/hub'))

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def metrics(y, probabilities):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), 5) or not np.isfinite(p).all():
        raise ValueError('Invalid probability matrix')
    prediction = p.argmax(axis=1)
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, prediction), 1)
    tp = cm.diagonal()
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    accuracy = float((prediction == y).mean())
    n = len(y)
    z = 1.959963984540054
    centre = (accuracy + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {'n': n, 'correct': int((prediction == y).sum()), 'accuracy': accuracy,
            'macro_f1': float(f1.mean()), 'nll': float(-np.log(p[np.arange(n), y].clip(1e-9, 1)).mean()),
            'accuracy_wilson_95': [centre-half, centre+half], 'confusion_matrix': cm.tolist(),
            'per_class': {c:{'precision':float(precision[i]), 'recall':float(recall[i]),
                             'f1':float(f1[i]), 'support':int(cm[i].sum())} for i,c in enumerate(CLASSES)}}

def rank(result):
    return result['macro_f1'], -result['nll']

def make_splits(plan):
    records = json.loads((ROOT / 'original_train_manifest.json').read_text())
    if len(records) != 500 or set(r['label'] for r in records) != set(range(5)):
        raise ValueError('Expected original 500 labeled images')
    hashes = [r['sha256'] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Exact duplicate images found: group before splitting')
    rng = np.random.default_rng(plan['seed'])
    audit, development, assignments = [], [], {}
    for label in range(5):
        indexes = np.array([i for i,r in enumerate(records) if r['label'] == label])
        if len(indexes) != 100:
            raise ValueError('Each original class must contain exactly 100 images')
        rng.shuffle(indexes)
        n_audit = round(len(indexes) * plan['audit_fraction'])
        audit.extend(indexes[:n_audit].tolist())
        dev = indexes[n_audit:]
        development.extend(dev.tolist())
        for fold, chunk in enumerate(np.array_split(dev, plan['development_folds'])):
            for i in chunk:
                assignments[int(i)] = fold
    folds = []
    for fold in range(plan['development_folds']):
        folds.append({'train':sorted(i for i in development if assignments[i] != fold),
                      'validation':sorted(i for i in development if assignments[i] == fold)})
    splits = {'seed':plan['seed'], 'audit':sorted(audit), 'development':sorted(development),
              'folds':folds, 'exact_duplicate_count':0,
              'train_manifest_sha256': hashlib.sha256((ROOT/'original_train_manifest.json').read_bytes()).hexdigest()}
    assert not set(audit) & set(development)
    assert sorted(audit + development) == list(range(500))
    assert sorted(i for f in folds for i in f['validation']) == sorted(development)
    dump(ROOT/'splits.json', splits)
    return records, splits

def foreground_crop(image):
    """Fixed, label-independent green segmentation. No learned/test-selected threshold."""
    rgb = np.asarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r,g,b = [rgb[:,:,i].astype(np.int16) for i in range(3)]
    mask = ((hsv[:,:,0]>=25) & (hsv[:,:,0]<=100) & (hsv[:,:,1]>=30) &
            (hsv[:,:,2]>=20) & (2*g-r-b>12)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return image.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = max(12, int(areas.max()*0.01))
    kept = np.isin(components, np.flatnonzero(areas>=threshold)+1)
    ys,xs = np.where(kept)
    if len(xs) < 12:
        return image.copy()
    margin = round(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*0.15)
    bbox = (max(0,int(xs.min())-margin), max(0,int(ys.min())-margin),
            min(image.width,int(xs.max())+margin+1), min(image.height,int(ys.max())+margin+1))
    return image.crop(bbox)

def square_pad(image):
    colour = tuple(np.median(np.asarray(image).reshape(-1,3), axis=0).astype(int).tolist())
    side = max(image.size)
    result = Image.new('RGB', (side,side), colour)
    result.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return result

def prepare_training_cache(records):
    for i,r in enumerate(records):
        source = (ROOT / r['path']).resolve()
        if not source.is_relative_to(ROOT / 'train'):
            raise ValueError('Training input escaped allowlist')
        if hashlib.sha256(source.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Original training file differs from manifest')
        destinations = [ROOT/'cache/images'/view/f'{i:04d}.jpg' for view in ('raw','crop')]
        if all(p.exists() for p in destinations):
            continue
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((1400,1400), Image.Resampling.LANCZOS)
        for view,p in zip(('raw','crop'), destinations):
            p.parent.mkdir(parents=True, exist_ok=True)
            result = square_pad(image if view=='raw' else foreground_crop(image))
            result.thumbnail((768,768), Image.Resampling.LANCZOS)
            result.save(p, quality=97, subsampling=0)
    print('Training cache verified; no test directory accessed', flush=True)

class RandomQuarterTurn:
    def __call__(self, image):
        return image.rotate(random.choice((0,90,180,270)))

def image_transform(config, training):
    size = config['size']
    operations = []
    if training and config['augmentation']:
        operations += [T.RandomResizedCrop(size, scale=(0.8,1.0), ratio=(0.9,1.1),
                                           interpolation=T.InterpolationMode.BICUBIC),
                       RandomQuarterTurn(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.ColorJitter(brightness=.15, contrast=.15, saturation=.1, hue=.025)]
    else:
        operations += [T.Resize((size,size), interpolation=T.InterpolationMode.BICUBIC)]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(MEAN,STD)])

class TrainImages(Dataset):
    def __init__(self, records, indexes, config, training):
        self.labels = [records[i]['label'] for i in indexes]
        self.paths = [ROOT/'cache/images'/config['view']/f'{i:04d}.jpg' for i in indexes]
        self.images = [Image.open(p).convert('RGB') for p in self.paths]
        self.transform = image_transform(config, training)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]

def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

def loader(dataset, plan, training, seed):
    return DataLoader(dataset, batch_size=plan['batch_size'], shuffle=training,
                      num_workers=plan['workers'], pin_memory=True, persistent_workers=plan['workers']>0,
                      worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed))

class DinoClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        repo = ROOT/'vendor/dinov2'
        self.backbone = torch.hub.load(str(repo), 'dinov2_vitl14', source='local', pretrained=False)
        if config['pretrained']:
            path = ROOT/'cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'
            self.backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        width = self.backbone.embed_dim
        self.head = nn.Sequential(nn.LayerNorm(width*2), nn.Linear(width*2,256), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(256,5))
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in self.backbone.blocks[-config['unfreeze_blocks']:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True
    def forward(self, images):
        f = self.backbone.forward_features(images)
        x = torch.cat([f['x_norm_clstoken'],f['x_norm_patchtokens'].mean(1)],dim=1)
        return self.head(x)

def build_model(config):
    arch = config['architecture']
    if arch == 'dinov2_large':
        return DinoClassifier(config)
    if arch == 'convnext_base':
        model = models.convnext_base(weights=None, stochastic_depth_prob=.1)
        if config['pretrained']:
            state = models.ConvNeXt_Base_Weights.IMAGENET1K_V1.get_state_dict(progress=False, check_hash=True)
            model.load_state_dict(state, strict=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features,5)
        return model
    if arch == 'resnet18':
        if config['pretrained']:
            raise ValueError('Baseline must be from scratch')
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features,5)
        return model
    raise ValueError(arch)

def optimizer_for(model, config):
    groups = {}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        head = name.startswith('head.') or name.startswith('classifier.2.') or name.startswith('fc.')
        lr = config['head_lr'] if head else config['backbone_lr']
        decay = config['weight_decay'] if p.ndim>=2 else 0.0
        groups.setdefault((lr,decay), []).append(p)
    return torch.optim.AdamW([{'params':parameters, 'lr':lr, 'initial_lr':lr, 'weight_decay':decay}
                              for (lr,decay),parameters in groups.items()], fused=True)

class EMA:
    """Keep only trainable tensors and model buffers, sharing the frozen backbone."""
    def __init__(self, model):
        self.parameters = {name:p for name,p in model.named_parameters() if p.requires_grad}
        self.buffers = dict(model.named_buffers())
        self.averages = {name:p.detach().clone() for name,p in self.parameters.items()}
        self.average_buffers = {name:b.detach().clone() for name,b in self.buffers.items()}
        self.steps = 0
    @torch.no_grad()
    def update(self):
        self.steps += 1
        decay = min(.97, (1+self.steps)/(10+self.steps))
        for name,p in self.parameters.items():
            self.averages[name].lerp_(p.detach(), 1-decay)
        for name,b in self.buffers.items():
            self.average_buffers[name].copy_(b)
    @contextlib.contextmanager
    def apply(self):
        originals = {name:p.data for name,p in self.parameters.items()}
        buffer_originals = {name:b.data for name,b in self.buffers.items()}
        try:
            for name,p in self.parameters.items():
                p.data = self.averages[name]
            for name,b in self.buffers.items():
                b.data = self.average_buffers[name]
            yield
        finally:
            for name,p in self.parameters.items():
                p.data = originals[name]
            for name,b in self.buffers.items():
                b.data = buffer_originals[name]

@torch.inference_mode()
def probabilities(model, data_loader, tta=True):
    model.eval()
    outputs = []
    for images,_ in data_loader:
        images = images.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = model(images).float().softmax(1)
            if tta:
                p = (p + model(images.flip(-1)).float().softmax(1))/2
        outputs.append(p.cpu().numpy())
    return np.concatenate(outputs)

def checkpoint(model, config, epoch, filename):
    trainable_names = {name for name,p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    if config['architecture']=='dinov2_large':
        # The immutable generic backbone is stored once in the official cache.
        state = {name:value for name,value in state.items() if name in trainable_names}
    state = {name:value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu()
             for name,value in state.items()}
    torch.save({'config':config, 'epoch':epoch, 'state':state}, filename)

def restore(model, config, filename):
    ckpt = torch.load(filename, map_location='cpu', weights_only=True)
    if ckpt['config'] != config:
        raise ValueError('Checkpoint recipe mismatch')
    strict = config['architecture']!='dinov2_large'
    missing, unexpected = model.load_state_dict(ckpt['state'], strict=strict)
    if unexpected or (not strict and any(not n.startswith('backbone.') for n in missing)):
        raise ValueError('Invalid checkpoint keys')
    return ckpt
