import os
import hashlib
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_image(path):
    """RGB uint8 HxWx3 (format detected from file content)."""
    raw = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


def read_label(path):
    """Binary uint8 HxW, 1 = vessel."""
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def preprocess(img, clahe=True):
    """Optional CLAHE on the LAB lightness channel to boost thin-vessel contrast."""
    if clahe == 2:
        green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img[..., 1])
        return np.repeat(green[..., None], 3, axis=2)
    if not clahe:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def normalize(img):
    return ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def build_model(arch, encoder, pretrained=True, encoder_weights=None):
    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}[arch]
    kw = dict(encoder_name=encoder, encoder_weights=None,
              in_channels=3, classes=1)
    if arch == "unet":
        kw["decoder_attention_type"] = "scse"
    model = cls(**kw)
    if pretrained:
        if encoder == 'resnet34':
            weights = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet34-b627a593.pth',
                map_location='cpu', check_hash=True)
        elif encoder == 'efficientnet-b3':
            from safetensors.torch import load_file
            path = Path(encoder_weights or Path(__file__).parent/'weights/efficientnet-b3.safetensors')
            expected = 'd95605b0390229a5aa0431a4f8231f175a31ce7a86b7a75200b8702207245253'
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('EfficientNet-B3 ImageNet checkpoint hash mismatch')
            weights = load_file(str(path), device='cpu')
        else:
            raise ValueError(encoder)
        model.encoder.load_state_dict(weights)
    return model


@torch.no_grad()
def predict_prob(model, img, scale=1.0, tta=True, device="cuda"):
    """Full-image inference. img: preprocessed RGB uint8 at original size.
    Returns float32 HxW vessel probability at original size."""
    h, w = img.shape[:2]
    if scale != 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    sh, sw = img.shape[:2]
    ph, pw = (32 - sh % 32) % 32, (32 - sw % 32) % 32
    x = torch.from_numpy(normalize(img))[None].to(device)
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    flips = [[], [2], [3], [2, 3]] if tta else [[]]
    acc = 0
    with (torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()):
        for f in flips:
            xi = torch.flip(x, f) if f else x
            y = torch.sigmoid(model(xi).float())
            acc = acc + (torch.flip(y, f) if f else y)
    p = (acc / len(flips))[0, 0, :sh, :sw].cpu().numpy()
    if scale != 1.0:
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    return p.astype(np.float32)


def eval_transform(mask01):
    """Exactly what segmentation_to_csv.py does: 0/255 'L' image -> PIL bilinear 512 -> >0."""
    im = Image.fromarray((mask01 > 0).astype(np.uint8) * 255, mode="L")
    im = im.resize((512, 512), Image.Resampling.BILINEAR)
    return np.array(im) > 0


def resize_prob_512(p):
    """Resize a float probability map to 512x512 with the same PIL bilinear filter."""
    return np.array(Image.fromarray(p.astype(np.float32), mode="F").resize(
        (512, 512), Image.Resampling.BILINEAR))


def dice(a, b, eps=1e-7):
    a, b = a.astype(bool), b.astype(bool)
    return (2.0 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps)


def list_ids(d):
    return sorted([os.path.splitext(f)[0] for f in os.listdir(d)], key=int)
