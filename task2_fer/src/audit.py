from pathlib import Path
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fer_data" / "fer_data"
CLASSES = ("Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral")
SEED = 20261011

def read_pixels(path):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        if im.size != (48, 48) or im.mode != "L":
            raise ValueError(f"Invalid image {path}: {im.size}, {im.mode}")
        return np.asarray(im, dtype=np.uint8).copy()

def phash(arr):
    small = np.asarray(Image.fromarray(arr).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    coeff = dctn(small, norm="ortho")[:8, :8].reshape(-1)
    bits = coeff[1:] >= np.median(coeff[1:])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description='Prepare grouped training folds from training images only.')
    parser.add_argument('--train-dir',type=Path,default=DATA/'train')
    train_dir=parser.parse_args().train_dir
    (ROOT/"artifacts").mkdir(parents=True,exist_ok=True)
    (ROOT/"data").mkdir(parents=True,exist_ok=True)
    train_paths = [(p, label, name) for label, name in enumerate(CLASSES)
                   for p in sorted((train_dir / name).glob("*.jpg"))]
    assert len(train_paths) == 28275, len(train_paths)
    pixels = []
    rows = []
    exact = defaultdict(list)
    hashes = []
    for i, (path, label, name) in enumerate(train_paths):
        arr = read_pixels(path)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        h = phash(arr)
        pixels.append(arr)
        hashes.append(h)
        exact[digest].append(i)
        rows.append(dict(index=i, path=path.relative_to(train_dir.parent).as_posix(), label=label,
                         emotion=name, sha256=digest, phash=f"{h:016x}"))
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for members in exact.values():
        for j in members[1:]:
            union(members[0], j)

    buckets = defaultdict(list)
    near_pairs = 0
    for i, h in enumerate(hashes):
        candidates = set()
        for segment in range(4):
            key = (segment, (h >> (16 * segment)) & 65535)
            candidates.update(buckets[key])
        for j in candidates:
            if (h ^ hashes[j]).bit_count() > 3:
                continue
            diff = pixels[i].astype(np.float32) - pixels[j]
            if np.mean(diff * diff) <= 100.0 and np.mean(np.abs(diff)) <= 5.0:
                if find(i) != find(j):
                    near_pairs += 1
                union(i, j)
        for segment in range(4):
            buckets[(segment, (h >> (16 * segment)) & 65535)].append(i)

    groups = [find(i) for i in range(len(rows))]
    labels = [r["label"] for r in rows]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = [-1] * len(rows)
    for fold, (_, valid) in enumerate(cv.split(np.zeros(len(rows)), labels, groups)):
        for i in valid:
            folds[i] = fold
    assert all(f >= 0 for f in folds)
    assert all(len({folds[i] for i in members}) == 1 for members in exact.values())
    for i, row in enumerate(rows):
        row["group"] = f"g{groups[i]:05d}"
        row["fold"] = folds[i]
    write_csv(ROOT / "artifacts" / "train_manifest.csv", rows,
              ["index", "path", "label", "emotion", "sha256", "phash", "group", "fold"])
    np.save(ROOT / "data" / "train_pixels.npy", np.stack(pixels))

    conflicts = sum(len({labels[i] for i in members}) > 1 for members in exact.values())
    result = dict(seed=SEED, class_map={c:i for i,c in enumerate(CLASSES)},
                  train_count=len(rows),
                  class_counts=dict(Counter(r["emotion"] for r in rows)),
                  exact_duplicate_groups=sum(len(v)>1 for v in exact.values()),
                  exact_duplicate_extra=sum(len(v)-1 for v in exact.values()),
                  conflicting_exact_groups=conflicts, near_group_merges=near_pairs,
                  fold_sizes=dict(Counter(folds)),
                  fold_class_counts={str(f):dict(Counter(rows[i]["emotion"] for i,x in enumerate(folds) if x==f))
                                     for f in range(5)})
    (ROOT / "artifacts" / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
