"""Training identities and submission transforms; no dataset paths on import."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def training_fingerprint(train_dir, ids):
    h=hashlib.sha256()
    for image_id in ids:
        for role in ("image","label"):
            p=Path(train_dir)/role/(str(image_id)+".jpg")
            h.update((role+"/"+str(image_id)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()

def check_warm_start(current, previous):
    for key in ("encoder","folds","full","training_fingerprint","validation_ids"):
        if key not in previous or previous[key] != current[key]:
            raise ValueError("Incompatible warm start: "+key)
    if not current.get("encoder_only_init") and current["arch"] != previous["arch"]:
        raise ValueError("Incompatible warm start: arch")

def resize_probability(prob):
    p=np.asarray(prob,dtype=np.float32)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError("Expected a finite probability image in [0,1]")
    return np.array(Image.fromarray(p).resize((512,512),Image.Resampling.BILINEAR))

def fuse(family_probabilities):
    """Prespecified equal family weights and threshold 0.5; no search state."""
    parts=[resize_probability(p) for p in family_probabilities]
    if len(parts)!=6:
        raise ValueError("Exactly six family probability maps are required")
    return np.mean(parts,axis=0)>.5

def rle_encode(mask):
    flat=np.asarray(mask,dtype=bool).reshape(-1).astype(np.uint8)
    padded=np.concatenate(([0],flat,[0]))
    changes=np.flatnonzero(padded[1:]!=padded[:-1])+1
    changes[1::2]-=changes[::2]
    return " ".join(map(str,changes.tolist()))

def rle_decode(encoded, shape=(512,512)):
    tokens=[int(x) for x in encoded.split()]
    if len(tokens)%2:
        raise ValueError("RLE must contain start/length pairs")
    flat=np.zeros(int(np.prod(shape)),dtype=bool);end=0
    for start,length in zip(tokens[::2],tokens[1::2]):
        if start<1 or length<1 or start-1<end or start-1+length>len(flat):
            raise ValueError("Invalid RLE interval")
        flat[start-1:start-1+length]=True;end=start-1+length
    return flat.reshape(shape)
