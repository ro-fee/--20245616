from __future__ import annotations


import os


import re


import gc


import json


import math


import random


import hashlib


import warnings


import platform


from pathlib import Path


from dataclasses import dataclass, asdict, replace


from collections import defaultdict


import cv2


import numpy as np


import pandas as pd


import matplotlib.pyplot as plt


from PIL import Image


import torch


from torch.utils.data import Dataset, DataLoader


from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass
class Config:
    # Usually leave DATA_DIR empty: locate the attached competition automatically.
    INPUT_ROOT: str = "/kaggle/input"
    DATA_DIR: str | None = None
    LABEL_CSV: str | None = None
    SAMPLE_CSV: str | None = None
    OUTPUT_DIR: str = "/kaggle/working/fovea_run"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    GROUP_COLUMN: str | None = None  # Optional real patient/subject column in labels.
    COORDINATE_ORIGIN: int = 0  # Change to 1 only if annotation documentation confirms it.
    DUPLICATE_LABEL_TOLERANCE_PX: float = 5.0  # Preserve small label differences; keep duplicates in one fold.
    SEED: int = 2026
    N_FOLDS: int = 5
    INPUT_SIZE: int = 512
    HEATMAP_SIZE: int = 128
    HEATMAP_SIGMA: float = 2.5
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    EPOCHS: int = 80
    PATIENCE: int = 12
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 3
    AMP: bool = True
    GRAD_CLIP: float = 1.0
    FREEZE_BN: bool = True
    COORD_SCALE: float = 512.0
    HEATMAP_LOSS_WEIGHT: float = 1.0
    COORD_LOSS_WEIGHT: float = 10.0
    VIS_LOSS_WEIGHT: float = 1.0
    PRETRAINED: bool = True
    ALLOW_WEIGHT_DOWNLOAD: bool = True
    PRETRAINED_WEIGHTS_PATH: str | None = None
    VISIBILITY_THRESHOLD: float = 0.5
    # False avoids reporting a threshold fitted on OOF labels as unbiased CV.
    TUNE_VISIBILITY_THRESHOLD: bool = False
    TTA: bool = True  # Original + horizontal flip; both decoded in original pixels.
    SHOW_PLOTS: bool = True
    REUSE_CHECKPOINTS: bool = False  # True only with identical data/config/code.
    SMOKE_TEST: bool = False  # 2 folds, 2 epochs, low resolution; NOT a scored model.


@dataclass
class ResearchConfig(Config):
    OUTPUT_DIR: str = "/kaggle/working/fovea_optimized"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    PRETRAINED: bool = True
    ALLOW_WEIGHT_DOWNLOAD: bool = False
    PRETRAINED_WEIGHTS_PATH: str | None = None
    N_FOLDS: int = 5
    CALIBRATION_FRACTION: float = 0.25
    EARLY_STOP_FRACTION: float = 0.20
    INNER_CROP_FOLDS: int = 3
    FINE_TRAIN_CROPS: str = "oof"  # "jitter" is a cheaper, explicitly different ablation.
    COARSE_EPOCHS: int = 50
    FINE_EPOCHS: int = 40
    PATIENCE: int = 8
    INPUT_SIZE: int = 512
    HEATMAP_SIZE: int = 128
    FINE_INPUT_SIZE: int = 384
    FINE_HEATMAP_SIZE: int = 96
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 2
    BACKBONE_LR_MULTIPLIER: float = 1.0  # 0.25 is an optional candidate, not a proven improvement.
    USE_EMA: bool = False  # Enable only after an equal-budget development comparison.
    EMA_DECAY: float = 0.99
    EMA_WARMUP_UPDATES: int = 10
    ROI_FRACTIONS: tuple = (0.15, 0.20, 0.30, 0.40, 0.60)
    FINE_MULTISCALE_TRAIN: bool = True  # Preserve OOF centre; sample the training field of view.
    SELECT_FINE_CROP_POLICY: bool = True  # Compare fixed/multiscale only on each fold's inner stopping split.
    FINE_MULTISCALE_EARLY_STOP: bool = True
    FINE_VALIDATION_ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)  # Prespecified inner-only mean RMSE.
    OOF_CENTER_JITTER: float = 0.08
    TRAIN_CENTER_JITTER: float = 0.20
    FIXED_ROI_FRACTION: float = 0.30
    ROI_MIN_FRACTION: float = 0.15
    ROI_MAX_FRACTION: float = 0.60
    ROI_MARGIN_PX: float = 16.0
    ROI_COVERAGE: float = 0.90
    RAW_RADIUS_MULTIPLIER: float = 2.0
    UNCERTAINTY_FLOOR_PX: float = 1.0
    FALLBACK_RULE_CANDIDATES: tuple = ((0.0, 2.0), (0.03, 4.0), (0.08, 1e9))
    FALLBACK_MIN_GAIN_PX: float = 1.0
    FALLBACK_MIN_RELATIVE_GAIN: float = 0.01
    FALLBACK_MIN_GROUPS: int = 12
    EXPERIMENTS: tuple = ("B0", "B1", "B2", "B3", "B4", "C1")
    SUBMISSION_EXPERIMENT: str = "B4"  # Preselected; never auto-pick the best outer OOF score.
    BOOTSTRAP_REPEATS: int = 1000
    REUSE_CHECKPOINTS: bool = True  # Reuse matching completed stages/folds after integrity checks.
    SHOW_PLOTS: bool = False
    SMOKE_TEST: bool = False
    SMOKE_FOLDS: int = 2
    SMOKE_INPUT_SIZE: int = 64
    SMOKE_EPOCHS: int = 1
    REQUIRE_GPU: bool = True


RESEARCH_VERSION = "3.0.0"


RESEARCH_WEIGHT_FILENAME = "resnet18-f37072fd.pth"


RESEARCH_WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


RESEARCH_WEIGHT_BYTES = 46830571


RESEARCH_WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def resolve_research_weights(cfg):
    """Return a SHA256-verified .pth path, or None only if PRETRAINED=False.

    PRETRAINED_WEIGHTS_PATH accepts a .pth, the supplied ZIP, or a directory.
    With no explicit path, recursively discover the official filename or
    fovea_pretrained_weights.zip under /kaggle/input. Kaggle may unpack uploaded
    ZIPs automatically; both layouts work. All matching copies must have the
    expected complete SHA256. A ZIP is streamed to a fixed filename under
    OUTPUT_DIR/pretrained_cache; input datasets are never modified.
    """
    from pathlib import Path, PurePosixPath
    import hashlib
    import os
    import urllib.request
    import zipfile

    if not bool(getattr(cfg, "PRETRAINED", True)):
        print("PRETRAINED=False: explicitly using random encoder initialization.")
        return None

    def checked_stream(stream, label, destination=None):
        checksum = hashlib.sha256()
        total = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            if total > RESEARCH_WEIGHT_BYTES:
                raise ValueError(f"Pretrained weight is larger than expected: {label}")
            checksum.update(block)
            if destination is not None:
                destination.write(block)
        actual = checksum.hexdigest()
        if total != RESEARCH_WEIGHT_BYTES or actual != RESEARCH_WEIGHT_SHA256:
            raise ValueError(
                f"Pretrained weight verification failed: {label}\n"
                f"Expected {RESEARCH_WEIGHT_BYTES} bytes / SHA256 {RESEARCH_WEIGHT_SHA256}; "
                f"got {total} bytes / SHA256 {actual}. "
                "Remove the corrupt or incompatible input and attach the supplied official bundle."
            )

    def check_file(path):
        with path.open("rb") as stream:
            checked_stream(stream, str(path))

    explicit = str(getattr(cfg, "PRETRAINED_WEIGHTS_PATH", "") or "").strip()
    root = Path(explicit).expanduser() if explicit else Path("/kaggle/input")
    if explicit and not root.exists():
        raise FileNotFoundError(f"PRETRAINED_WEIGHTS_PATH does not exist: {root}")
    if root.is_file():
        inputs = [root]
    elif root.is_dir():
        inputs = sorted(set(root.rglob(RESEARCH_WEIGHT_FILENAME)) |
                        set(root.rglob("fovea_pretrained_weights.zip")))
    else:
        inputs = []

    # Verify every matching source before choosing one: conflicting copies fail.
    candidates = []
    for path in inputs:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as archive:
                members = [info for info in archive.infolist()
                           if not info.is_dir() and
                           PurePosixPath(info.filename.replace("\\", "/")).name == RESEARCH_WEIGHT_FILENAME]
                if not members:
                    raise ValueError(f"No {RESEARCH_WEIGHT_FILENAME} found inside {path}")
                for info in members:
                    entry = PurePosixPath(info.filename.replace("\\", "/"))
                    if entry.is_absolute() or ".." in entry.parts:
                        raise ValueError(f"Unsafe ZIP member name: {info.filename}")
                    if info.file_size != RESEARCH_WEIGHT_BYTES:
                        raise ValueError(f"Wrong pretrained member size: {path}!{info.filename}")
                    with archive.open(info, "r") as stream:
                        checked_stream(stream, f"{path}!{info.filename}")
                    candidates.append((path, info.filename))
        else:
            check_file(path)
            candidates.append((path, None))

    if candidates:
        # Prefer already-unpacked input when duplicate copies are identical.
        plain = [path for path, member in candidates if member is None]
        if plain:
            selected = plain[0].resolve()
            print(f"Verified official ImageNet-1K ResNet18 weights: {selected}")
            return str(selected)

    if not candidates and explicit:
        raise FileNotFoundError(
            f"No {RESEARCH_WEIGHT_FILENAME} or fovea_pretrained_weights.zip found under {root}."
        )

    cache_dir = (Path(cfg.OUTPUT_DIR).expanduser() / "pretrained_cache").resolve()
    kaggle_input = Path("/kaggle/input").resolve()
    if cache_dir == kaggle_input or kaggle_input in cache_dir.parents:
        raise ValueError("OUTPUT_DIR must be writable and outside /kaggle/input.")
    cache = cache_dir / RESEARCH_WEIGHT_FILENAME
    if cache.exists():
        check_file(cache)
        print(f"Verified pretrained weight cache: {cache}")
        return str(cache)

    if not candidates and not bool(getattr(cfg, "ALLOW_WEIGHT_DOWNLOAD", False)):
        raise FileNotFoundError(
            "Pretrained weights were not found. Upload fovea_pretrained_weights.zip as a Kaggle "
            "Dataset and attach it with Add Input, or set PRETRAINED_WEIGHTS_PATH to its .pth, "
            "ZIP, or folder. For an online download, explicitly set ALLOW_WEIGHT_DOWNLOAD=True "
            "and enable Notebook Internet. No random-initialization fallback is performed."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f"{RESEARCH_WEIGHT_FILENAME}.{os.getpid()}.partial"
    try:
        if candidates:
            archive_path, member = candidates[0]
            with zipfile.ZipFile(archive_path, "r") as archive:
                with archive.open(member, "r") as stream, temporary.open("wb") as output:
                    checked_stream(stream, f"{archive_path}!{member}", output)
        else:
            print(f"Downloading official ImageNet-1K weights: {RESEARCH_WEIGHT_URL}")
            request = urllib.request.Request(RESEARCH_WEIGHT_URL, headers={"User-Agent": "fovea-research-notebook/1.0"})
            with urllib.request.urlopen(request, timeout=30) as stream, temporary.open("wb") as output:
                checked_stream(stream, RESEARCH_WEIGHT_URL, output)
        temporary.replace(cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Verified pretrained weights available at {cache}")
    return str(cache)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: some CUDA kernels do not have deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(0)


def csv_read(path):
    # Strings preserve leading zeroes in image names and exact submission IDs.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate CSV columns: {path}")
    return frame


def column_name(frame, expected):
    matches = [c for c in frame.columns if c.casefold() == expected.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected column {expected!r}; found {list(frame.columns)}")
    return matches[0]


def discover_data(cfg):
    base = Path(cfg.DATA_DIR or cfg.INPUT_ROOT)
    if not base.exists():
        raise FileNotFoundError(
            f"{base} does not exist. In Kaggle, Add Input -> Competition -> 2026task3, "
            "then set CFG.DATA_DIR if automatic discovery is ambiguous."
        )
    csvs = sorted(base.rglob("*.csv"))
    if cfg.LABEL_CSV:
        label_path = Path(cfg.LABEL_CSV)
    else:
        candidates = [p for p in csvs if p.name.casefold() == "fovea_localization_train_gt.csv"]
        if not candidates:
            candidates = []
            for p in csvs:
                try:
                    cols = set(pd.read_csv(p, nrows=0).columns.str.strip().str.casefold())
                    if {"data", "fovea_x", "fovea_y"} <= cols:
                        candidates.append(p)
                except Exception:
                    pass
        if len(candidates) != 1:
            raise ValueError(
                "Cannot uniquely locate labels. Set CFG.DATA_DIR or CFG.LABEL_CSV. "
                f"Candidates: {[str(p) for p in candidates]}"
            )
        label_path = candidates[0]
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    if cfg.SAMPLE_CSV:
        sample_path = Path(cfg.SAMPLE_CSV)
        root = base if cfg.DATA_DIR else Path(os.path.commonpath([label_path.parent, sample_path.parent]))
    else:
        # Search nearest enclosing directory first, avoiding other attached datasets.
        root = label_path.parent
        while True:
            candidates = [p for p in root.rglob("*.csv") if p.name.casefold() == "sample_submission.csv"]
            if candidates:
                if len(candidates) != 1:
                    raise ValueError(f"Several sample submissions under {root}; set CFG.SAMPLE_CSV.")
                sample_path = candidates[0]
                break
            if root == base or root.parent == root or base not in root.parents:
                raise FileNotFoundError("sample_submission.csv missing. Set CFG.SAMPLE_CSV explicitly.")
            root = root.parent
        if cfg.DATA_DIR:
            root = base
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)
    images = sorted(p for p in root.rglob("*") if p.suffix.casefold() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(
            f"No extracted images under {root}. Add the competition data in Kaggle; "
            "if your own upload contains only a ZIP, extract it and set CFG.DATA_DIR."
        )
    print(f"Data root: {root}\nLabels: {label_path}\nSample: {sample_path}\nImage files: {len(images)}")
    print("Image path examples:", [str(p.relative_to(root)) for p in images[:12]])
    return root, label_path, sample_path, images


def canonical_image_id(value):
    name = str(value).strip().replace("\\", "/").split("/")[-1]
    return Path(name).stem.casefold() if Path(name).suffix.casefold() in IMAGE_EXTENSIONS else name.casefold()


def parse_submission_id(value):
    text = str(value)
    # Remove only a recognized suffix; retain underscores and leading zeroes in image names.
    match = re.search(r"(?i)(?:[_\-\s]+)?fovea[_\-\s]*([xy])\s*$", text)
    if match is None:
        match = re.search(r"(?i)[_\-\s]+([xy])\s*$", text)
    if match is None:
        raise ValueError(f"Unrecognized ImageID {text!r}; expected image name + Fovea_X/Y.")
    name = text[:match.start()].rstrip("_- ")
    if not name:
        raise ValueError(f"Empty image name in ImageID {text!r}")
    return canonical_image_id(name), match.group(1).lower()


class ImageIndex(defaultdict):
    def __init__(self, root=None):
        super().__init__(list)
        self.root = Path(root) if root is not None else None


def build_image_index(paths, root=None):
    index = ImageIndex(root)
    for path in paths:
        path = Path(path)
        index[canonical_image_id(path.name)].append(path)
    return index


def image_split_role(path, root=None):
    """Use the nearest explicit split directory, limited to the dataset root."""
    path = Path(path)
    if root is not None:
        try:
            parts = path.relative_to(root).parts[:-1]
        except ValueError:
            raise ValueError(f"Image path {path} is outside the selected dataset root {root}.")
    else:
        parts = path.parts[:-1]
    for part in reversed(parts):
        tokens = set(re.split(r"[^a-z0-9]+", part.casefold()))
        is_train = bool(tokens & {"train", "training"})
        is_test = bool(tokens & {"test", "testing"})
        if is_train or is_test:
            return "ambiguous" if is_train and is_test else ("train" if is_train else "test")
    filename_role = re.match(r"^(train(?:ing)?|test(?:ing)?)[_\-\s.]", path.stem.casefold())
    if filename_role:
        return "train" if filename_role.group(1).startswith("train") else "test"
    return None


def numeric_image_signature(value, csv_identifier=False):
    """Single numeric block only: never extract a patient/date's trailing number."""
    key = canonical_image_id(value)
    if csv_identifier and re.fullmatch(r"[0-9]+\.0+", key):
        key = key.split(".")[0]  # Spreadsheet-exported integer, e.g. '1.0'.
    match = re.fullmatch(r"([^0-9]*)([0-9]+)([^0-9]*)", key)
    if match is None:
        return None
    prefix, number, suffix = match.groups()
    return prefix, number.lstrip("0") or "0", suffix


def resolve_image(index, image_id, role):
    if role not in {"train", "test"}:
        raise ValueError(f"Unknown image role: {role}")
    root = getattr(index, "root", None)
    all_paths = list(dict.fromkeys(path for paths in index.values() for path in paths))
    roles = {path: image_split_role(path, root) for path in all_paths}

    def scope(paths):
        # Never use an explicitly opposite-split image, even if its name is exact.
        eligible = [path for path in paths if roles[path] in (None, role)]
        explicit = [path for path in eligible if roles[path] == role]
        return explicit if explicit else eligible

    key = canonical_image_id(image_id)
    matches = scope(index.get(key, []))
    stage = "exact filename"
    if not matches:
        signature = numeric_image_signature(image_id, csv_identifier=True)
        if signature is not None:
            prefix, number, suffix = signature
            eligible = scope(all_paths)
            signatures = {path: numeric_image_signature(path.name) for path in eligible}
            # Prefer the same textual prefix/suffix: '01' -> '001', 'eye_1' -> 'eye_001'.
            matches = [path for path, item in signatures.items() if item == (prefix, number, suffix)]
            stage = "same prefix/suffix with normalized numeric ID"
            if not matches and prefix == "" and suffix == "":
                # CSV '1' may identify 'image_001.jpg'. Require a unique single-number
                # filename in this split; 'patient12_eye1' is deliberately not guessed.
                matches = [path for path, item in signatures.items() if item is not None and item[1] == number]
                stage = "unique numeric ID in a single-number filename"
    if len(matches) != 1:
        examples = [str(path.relative_to(root) if root is not None else path) for path in scope(all_paths)[:12]]
        raise ValueError(
            f"{role} image {image_id!r}: expected one unambiguous file at {stage}, "
            f"found {[str(p) for p in matches]}. Available {role} examples: {examples}. "
            "Check the label/file naming convention; no positional or substring matching is used."
        )
    return matches[0]


def read_rgb(path):
    # PIL supports non-ASCII local paths; do not apply EXIF rotations without changing labels.
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def prepare_tables(cfg):
    root, label_path, sample_path, images = discover_data(cfg)
    labels, sample = csv_read(label_path), csv_read(sample_path)
    dc, xc, yc = [column_name(labels, n) for n in ("data", "Fovea_X", "Fovea_Y")]
    ic, vc = [column_name(sample, n) for n in ("ImageID", "value")]
    if list(sample.columns) != [ic, vc]:
        raise ValueError("Expected sample_submission to contain exactly ImageID and value.")
    index = build_image_index(images, root=root)
    train = pd.DataFrame({
        "image_id": labels[dc].map(canonical_image_id),
        "source_id": labels[dc],
        "x": pd.to_numeric(labels[xc], errors="raise"),
        "y": pd.to_numeric(labels[yc], errors="raise"),
    })
    if not len(train) or train.image_id.duplicated().any():
        raise ValueError("Training labels are empty or contain repeated image IDs.")
    if not np.isfinite(train[["x", "y"]].to_numpy(float)).all():
        raise ValueError("Training coordinates contain NaN or infinity.")
    train["visible"] = ~((train.x == 0) & (train.y == 0))
    # Internally all image coordinates are zero-based pixel centres. Preserve sentinel.
    train.loc[train.visible, ["x", "y"]] -= cfg.COORDINATE_ORIGIN
    train["path"] = [str(resolve_image(index, name, "train")) for name in train.source_id]
    if train.path.duplicated().any():
        raise ValueError(f"Different training IDs map to the same file:\n{train.loc[train.path.duplicated(False), ['source_id', 'path']]}")

    if sample[ic].duplicated().any():
        raise ValueError("Duplicate submission ImageIDs.")
    parsed = [parse_submission_id(v) for v in sample[ic]]
    layout = pd.DataFrame(parsed, columns=["image_id", "axis"])
    counts = layout.groupby(["image_id", "axis"]).size().unstack(fill_value=0)
    if set(counts.columns) != {"x", "y"} or not (counts[["x", "y"]] == 1).all().all():
        raise ValueError("Each test image must have exactly one X row and one Y row.")
    test_ids = list(dict.fromkeys(layout.image_id))
    test = pd.DataFrame({"image_id": test_ids})
    test["path"] = [str(resolve_image(index, name, "test")) for name in test_ids]
    if test.path.duplicated().any():
        raise ValueError(f"Different test IDs map to the same file:\n{test.loc[test.path.duplicated(False), ['image_id', 'path']]}")
    if set(train.path) & set(test.path):
        raise ValueError("A physical image is referenced by both train and test. Check ID mapping.")
    print("Training ID -> file examples:", list(zip(train.source_id.head(8), train.path.map(lambda p: Path(p).name).head(8))))
    print("Test ID -> file examples:", list(zip(test.image_id.head(8), test.path.map(lambda p: Path(p).name).head(8))))

    for frame, role in ((train, "train"), (test, "test")):
        sizes = []
        hashes = []
        for path in frame.path:
            rgb = read_rgb(path)
            h, w = rgb.shape[:2]
            if min(w, h) < 2:
                raise ValueError(f"Invalid image dimensions: {path}")
            sizes.append((w, h))
            hashes.append(hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest())
        frame[["width", "height"]] = pd.DataFrame(sizes, index=frame.index)
        frame["pixel_hash"] = hashes
        if role == "train":
            bad = frame.visible & ((frame.x < 0) | (frame.y < 0) |
                                   (frame.x > frame.width - 1) | (frame.y > frame.height - 1))
            if bad.any():
                raise ValueError(f"Visible labels outside zero-based image bounds:\n{frame.loc[bad]}")
    if set(train.pixel_hash) & set(test.pixel_hash):
        warnings.warn("Exact pixel duplicates exist between train and test; inspect dataset provenance.")
    if not train.visible.any():
        raise ValueError("All training labels are (0,0): cannot train a supervised localization model.")

    group_column = cfg.GROUP_COLUMN
    if group_column is None:
        group_column = next((c for c in labels if c.casefold() in {"patient_id", "subject_id", "patient", "subject"}), None)
    if group_column is not None:
        if group_column not in labels:
            raise ValueError(f"GROUP_COLUMN {group_column!r} missing from label CSV.")
        raw_groups = labels[group_column].str.strip()
        if (raw_groups == "").any():
            raise ValueError("Patient group column contains empty values.")
        train["patient_group"] = raw_groups
    else:
        train["patient_group"] = train.image_id
        warnings.warn("No patient IDs available: grouping exact duplicates only; patient independence is unverified.")
    train["group"] = connected_groups(train)
    duplicate_audit = []
    for pixel_hash, group in train.groupby("pixel_hash"):
        if len(group) < 2:
            continue
        points = group[["x", "y"]].to_numpy(float)
        maximum_distance = float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).max())
        consistent_visibility = group.visible.nunique() == 1
        ids = group.image_id.tolist()
        duplicate_audit.append({
            "pixel_hash": pixel_hash,
            "image_ids": ",".join(ids),
            "count": len(group),
            "max_label_distance_px": maximum_distance,
            "visibility_consistent": consistent_visibility,
            "policy": "keep original rows and coordinates; force same validation fold",
        })
        if not consistent_visibility:
            raise ValueError(f"Identical pixels have conflicting visible/invisible labels: {ids}")
        if maximum_distance > cfg.DUPLICATE_LABEL_TOLERANCE_PX:
            raise ValueError(
                f"Identical pixels {ids} have label differences of {maximum_distance:.3f}px, "
                f"above DUPLICATE_LABEL_TOLERANCE_PX={cfg.DUPLICATE_LABEL_TOLERANCE_PX}. "
                "Inspect the annotations before changing the tolerance."
            )
        if maximum_distance > 0:
            warnings.warn(
                f"Duplicate images {ids} differ in point annotation by {maximum_distance:.3f}px. "
                "Keeping both ORIGINAL labels and placing them in the SAME fold; see duplicate_label_audit.csv."
            )
    print(f"Train={len(train)}, test={len(test)}, submission rows={len(sample)}")
    print(f"Visible={int(train.visible.sum())}, invisible={int((~train.visible).sum())}, groups={train.group.nunique()}")
    if len(train) != 80 or len(test) != 20:
        warnings.warn("Counts differ from the published 80 train / 20 test. Verify the selected data.")
    return train, test, sample, layout, {"root": str(root), "label_csv": str(label_path),
                                      "sample_csv": str(sample_path), "id_column": ic, "value_column": vc,
                                      "duplicate_label_audit": duplicate_audit}


def connected_groups(frame):
    # Union both patient and duplicate relationships; neither may cross a fold.
    parent = list(range(len(frame)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    for column in ("patient_group", "pixel_hash"):
        first = {}
        for i, key in enumerate(frame[column]):
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    return [f"group_{find(i):04d}" for i in range(len(frame))]


def assign_folds(train, cfg):
    frame = train.copy()
    n_folds = min(cfg.N_FOLDS, frame.group.nunique())
    if n_folds < 2:
        raise ValueError("At least two independent groups are required for cross-validation.")
    class_groups = frame.groupby("visible").group.nunique()
    if len(class_groups) == 2 and class_groups.min() >= 2:
        n_folds = min(n_folds, int(class_groups.min()))
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.SEED)
        splits = splitter.split(frame, frame.visible.astype(int), frame.group)
    else:
        # Randomize group labels before deterministic GroupKFold assignment.
        names = frame.group.unique().copy()
        np.random.default_rng(cfg.SEED).shuffle(names)
        group_numbers = frame.group.map({name: i for i, name in enumerate(names)})
        splits = GroupKFold(n_splits=n_folds).split(frame, groups=group_numbers)
        if len(class_groups) == 2:
            warnings.warn("Too few independent invisible/visible groups for stratification; visibility CV is unreliable.")
    frame["fold"] = -1
    for fold, (tr, va) in enumerate(splits):
        if set(frame.iloc[tr].group) & set(frame.iloc[va].group):
            raise AssertionError("Group leakage in fold assignment.")
        if not frame.iloc[tr].visible.any():
            raise ValueError("A training fold has no visible examples; reduce folds or inspect group labels.")
        frame.loc[frame.index[va], "fold"] = fold
    assert (frame.fold >= 0).all()
    print(frame.groupby("fold").agg(images=("image_id", "size"), visible=("visible", "sum")))
    return frame


def letterbox_geometry(width, height, size):
    factor = min(size / width, size / height)
    new_width = min(size, max(1, int(round(width * factor))))
    new_height = min(size, max(1, int(round(height * factor))))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    sx, sy = new_width / width, new_height / height
    # OpenCV resize maps pixel centres: dst = scale*(src+0.5)-0.5.
    transform = np.array([[sx, 0, left + (sx - 1) / 2],
                          [0, sy, top + (sy - 1) / 2],
                          [0, 0, 1]], dtype=np.float64)
    return new_width, new_height, left, top, transform


def letterbox(image, size):
    height, width = image.shape[:2]
    nw, nh, left, top, transform = letterbox_geometry(width, height, size)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if nw < width else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, transform


def gaussian_heatmap(xy, size, sigma):
    x, y = np.asarray(xy, dtype=np.float32) * (size - 1)
    axis = np.arange(size, dtype=np.float32)
    values = np.exp(-((axis[None, :] - x)**2 + (axis[:, None] - y)**2) / (2 * sigma**2))
    values /= max(float(values.sum()), 1e-12)
    return values[None].astype(np.float32)


import socket


import time


from urllib.parse import urlparse


from torch import nn


from torch.nn import functional as F


_RESNET18_STATE = None


_RESNET18_DOWNLOAD_FAILED = False


def _load_torch_file(path, map_location="cpu"):
    # Our checkpoints contain tensors and primitive values, not pickled modules.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch installations.
        return torch.load(path, map_location=map_location)


def build_resnet18(pretrained=True, allow_download=True, weights_path=None):
    """Use explicit local weights/cache first; Internet is optional.

    Kaggle Internet=Off users can attach a dataset containing the official
    resnet18-f37072fd.pth and supply its path as weights_path. A failed download
    produces a clearly reported random-initialization fallback.
    """
    global _RESNET18_STATE, _RESNET18_DOWNLOAD_FAILED
    from torchvision.models import ResNet18_Weights, resnet18

    encoder = resnet18(weights=None)
    if not pretrained:
        return encoder
    if weights_path:
        explicit_path = Path(weights_path)
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Pretrained weights do not exist: {explicit_path}")
        state = _load_torch_file(explicit_path)
        encoder.load_state_dict(state, strict=True)
        return encoder
    if _RESNET18_STATE is not None:
        encoder.load_state_dict(_RESNET18_STATE, strict=True)
        return encoder

    weights = ResNet18_Weights.DEFAULT
    cache_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    )
    try:
        if cache_path.is_file():
            state = _load_torch_file(cache_path)
        elif allow_download and not _RESNET18_DOWNLOAD_FAILED:
            # torchvision delegates to urllib without exposing its timeout.
            # Restrict only this synchronous download; restore the caller's
            # socket default even when a timeout or checksum error is raised.
            previous_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(20.0)
                state = weights.get_state_dict(progress=True, check_hash=True)
            finally:
                socket.setdefaulttimeout(previous_timeout)
        else:
            reason = "downloads disabled" if not allow_download else "previous download failed"
            warnings.warn(
                f"ResNet18 pretrained weights unavailable ({reason}); "
                "using random initialization. This may reduce accuracy on 80 images."
            )
            return encoder
        encoder.load_state_dict(state, strict=True)
        _RESNET18_STATE = state
    except Exception as exc:
        _RESNET18_DOWNLOAD_FAILED = True
        warnings.warn(
            f"Could not load ResNet18 pretrained weights: {type(exc).__name__}: {exc}. "
            "Using random initialization. Attach official weights for offline use."
        )
    return encoder


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvGNAct(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


def heatmap_to_coords(logits):
    """Spatial softmax expectation in [0,1], retaining subpixel coordinates.

    A multi-peak heatmap can place its expectation between peaks; inspect OOF
    errors before changing the decoder or introducing a refinement model.
    """
    batch_size, _, height, width = logits.shape
    probabilities = logits.float().flatten(1).softmax(dim=1).reshape(
        batch_size, height, width
    )
    xs = torch.linspace(0, 1, width, device=logits.device, dtype=torch.float32)
    ys = torch.linspace(0, 1, height, device=logits.device, dtype=torch.float32)
    x = (probabilities.sum(dim=1) * xs).sum(dim=1)
    y = (probabilities.sum(dim=2) * ys).sum(dim=1)
    return torch.stack([x, y], dim=1)


class FoveaNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        allow_download=True,
        use_visibility=True,
        heatmap_size=128,
        freeze_bn=True,
        pretrained_weights_path=None,
    ):
        super().__init__()
        self.encoder = build_resnet18(
            pretrained=pretrained,
            allow_download=allow_download,
            weights_path=pretrained_weights_path,
        )
        self.encoder.fc = nn.Identity()
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.heatmap_head = nn.Conv2d(64, 1, kernel_size=1)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )
        self.use_visibility = bool(use_visibility)
        self.freeze_bn = bool(freeze_bn)
        self.heatmap_size = int(heatmap_size)
        if not self.use_visibility:
            for parameter in self.visibility_head.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            # Small folds/batches give poor running-statistic estimates.
            # Affine BN weights still receive gradients.
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        e = self.encoder
        x = e.relu(e.bn1(e.conv1(image)))
        c1 = e.layer1(e.maxpool(x))
        c2 = e.layer2(c1)
        c3 = e.layer3(c2)
        c4 = e.layer4(c3)
        decoded = self.decoder3(c4, c3)
        decoded = self.decoder2(decoded, c2)
        decoded = self.decoder1(decoded, c1)
        logits = self.heatmap_head(decoded)
        if logits.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            logits = F.interpolate(
                logits,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.use_visibility:
            visibility_logits = self.visibility_head(c4).squeeze(1)
        else:
            # Disabled only when the training set is all visible. The data
            # validation/orchestrator rejects an all-invisible training fold.
            visibility_logits = logits.new_zeros(image.shape[0])
        return {
            "logits": logits,
            "coords": heatmap_to_coords(logits),
            "visibility_logits": visibility_logits,
        }


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def fovea_loss(output, batch, cfg, use_visibility):
    """Mean KL + original-pixel coordinate MSE + optional visibility BCE."""
    target_heatmap = batch["target_heatmap"].float()
    if target_heatmap.ndim == 3:
        target_heatmap = target_heatmap.unsqueeze(1)
    if target_heatmap.shape != output["logits"].shape:
        raise ValueError(
            f"Heatmap shape mismatch: target={target_heatmap.shape}, "
            f"prediction={output['logits'].shape}"
        )
    visible = batch["visible"].float().reshape(-1)
    n_visible = visible.sum().clamp_min(1.0)
    target_prob = target_heatmap.flatten(1)
    target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = output["logits"].float().flatten(1).log_softmax(dim=1)
    per_image_kl = (
        target_prob * (target_prob.clamp_min(1e-12).log() - log_prob)
    ).sum(dim=1)
    heatmap_loss = (per_image_kl * visible).sum() / n_visible

    # Convert input-coordinate differences to original pixels using the same
    # affine geometry as inference. No division by each image's own dimensions:
    # a 20-pixel error has the same loss on small and large originals.
    image_height, image_width = batch["image"].shape[-2:]
    input_extent = output["coords"].new_tensor([image_width - 1, image_height - 1])
    delta_input = (output["coords"] - batch["target_xy"].float()) * input_extent
    inverse_linear = batch["input_to_orig"].float()[:, :, :2]
    delta_original = torch.bmm(inverse_linear, delta_input.unsqueeze(-1)).squeeze(-1)
    per_image_coordinate = (
        delta_original / float(getattr(cfg, "COORD_SCALE", 512.0))
    ).square().mean(dim=1)
    coordinate_loss = (per_image_coordinate * visible).sum() / n_visible

    if use_visibility:
        visibility_loss = F.binary_cross_entropy_with_logits(
            output["visibility_logits"].float(), visible
        )
    else:
        visibility_loss = output["logits"].float().sum() * 0.0
    total = (
        float(getattr(cfg, "HEATMAP_LOSS_WEIGHT", 1.0)) * heatmap_loss
        + float(getattr(cfg, "COORD_LOSS_WEIGHT", 10.0)) * coordinate_loss
        + float(getattr(cfg, "VIS_LOSS_WEIGHT", 1.0)) * visibility_loss
    )
    return total, {
        "heatmap": float(heatmap_loss.detach()),
        "coordinate": float(coordinate_loss.detach()),
        "visibility": float(visibility_loss.detach()),
    }


def _autocast_context(device, cfg):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=bool(getattr(cfg, "AMP", True)) and device.type == "cuda",
    )


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = _load_torch_file(checkpoint_path, map_location="cpu")
    if bool(checkpoint.get("use_visibility", model.use_visibility)) != model.use_visibility:
        raise ValueError("Checkpoint and model use_visibility settings differ.")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return checkpoint


import copy


def research_crop_transform(center_x, center_y, side, input_size):
    """Map original pixel centres into a square crop with pixel-edge side `side`.

    The crop covers [cx-side/2, cx+side/2] in pixel-edge coordinates relative
    to its centre. Output pixel centres are symmetrically located around
    (input_size-1)/2. cv2.warpAffine receives this forward transform and performs
    inverse sampling itself. Crops may extend beyond the original image.
    """
    side = float(side)
    if not np.isfinite([center_x, center_y, side]).all() or side <= 0:
        raise ValueError("Crop centre and side must be finite; side must be positive.")
    scale = float(input_size) / side
    centre = (float(input_size) - 1.0) / 2.0
    return np.array([[scale, 0.0, centre - scale * float(center_x)],
                     [0.0, scale, centre - scale * float(center_y)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def research_roi_side(short_side, fraction, cfg):
    """Same base-fraction + 2*margin and clamp used by make_roi_spec."""
    short_side = np.asarray(short_side, dtype=np.float64)
    fraction = float(fraction)
    low, high = float(cfg.ROI_MIN_FRACTION), float(cfg.ROI_MAX_FRACTION)
    margin = float(cfg.ROI_MARGIN_PX)
    if not np.isfinite(short_side).all() or (short_side <= 1).any():
        raise ValueError("Image short sides must be finite and greater than one pixel.")
    if not math.isfinite(fraction) or fraction <= 0 or not 0 < low <= high <= 1:
        raise ValueError("Invalid ROI sampling fraction or fraction bounds.")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("ROI margin must be finite and nonnegative.")
    return np.clip(fraction * short_side + 2 * margin, low * short_side, high * short_side)


class ResearchDataset(Dataset):
    """All-visible localization with optional prediction-centred fine crops.

    Fine training without crop_spec uses random GT-centred perturbations. This
    is a declared synthetic localization-error augmentation, not inner OOF
    predictions. Validation/inference never repairs a crop using the label.
    """

    def __init__(self, frame, cfg, training=False, stage="coarse", crop_spec=None):
        if stage not in {"coarse", "fine"}:
            raise ValueError(f"Unknown stage: {stage}")
        self.frame = frame.reset_index(drop=True).copy()
        self.cfg, self.training, self.stage = cfg, bool(training), stage
        self.has_labels = {"x", "y", "visible"}.issubset(self.frame.columns)
        if self.training and not self.has_labels:
            raise ValueError("Training requires x, y and visible labels.")
        if self.has_labels and not self.frame.visible.astype(bool).all():
            raise ValueError("Research model supports this all-visible dataset only.")
        self.size = int(cfg.INPUT_SIZE if stage == "coarse" else cfg.FINE_INPUT_SIZE)
        self.hm_size = int(cfg.HEATMAP_SIZE if stage == "coarse" else cfg.FINE_HEATMAP_SIZE)
        self.crops = None
        if crop_spec is not None:
            required = ["image_id", "crop_x", "crop_y", "crop_side"]
            if not set(required).issubset(crop_spec.columns):
                raise ValueError(f"crop_spec must contain {required}")
            self.crops = crop_spec[required].copy()
            self.crops["image_id"] = self.crops.image_id.astype(str)
            if self.crops.image_id.duplicated().any():
                raise ValueError("crop_spec contains duplicate image IDs.")
            self.crops = self.crops.set_index("image_id")
            missing = set(self.frame.image_id.astype(str)) - set(self.crops.index)
            if missing:
                raise ValueError(f"Missing crops for image IDs: {sorted(missing)}")
            values = self.crops[["crop_x", "crop_y", "crop_side"]].to_numpy(float)
            if not np.isfinite(values).all() or (values[:, 2] <= 0).any():
                raise ValueError("Non-finite crop coordinates or non-positive crop side.")
        if stage == "fine" and not self.training and self.crops is None:
            raise ValueError("Fine validation/inference requires prediction-derived crop_spec.")
        self.roi_fractions = np.asarray(getattr(cfg, "ROI_FRACTIONS", (0.2, 0.3, 0.4)), float)
        if self.roi_fractions.size == 0 or not np.isfinite(self.roi_fractions).all() or (self.roi_fractions <= 0).any():
            raise ValueError("ROI_FRACTIONS must be finite positive numbers.")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        if self.stage == "coarse":
            image, cached_transform = optimized_coarse_view(row.path, self.size, int(row.width), int(row.height))
            height, width = int(row.height), int(row.width)
        else:
            image = read_rgb(row.path)
            height, width = image.shape[:2]
        if int(row.width) != width or int(row.height) != height:
            raise ValueError(f"Image dimensions changed: {row.path}")
        original = np.array([float(row.x), float(row.y)], np.float64) if self.has_labels else None
        crop_was_enlarged, initial_crop_miss = False, False
        sampled_fraction = -1.0  # Negative means no random scale was selected.
        crop_from_prediction = self.stage == "fine" and self.crops is not None
        if self.stage == "coarse":
            transform = cached_transform
            crop = np.array([(width - 1) / 2, (height - 1) / 2, max(width, height)], float)
        else:
            if self.crops is not None:
                crop = self.crops.loc[str(row.image_id), ["crop_x", "crop_y", "crop_side"]].to_numpy(float).copy()
                if self.training:
                    if bool(getattr(self.cfg, "FINE_MULTISCALE_TRAIN", True)):
                        sampled_fraction = float(np.random.choice(self.roi_fractions))
                        crop[2] = float(research_roi_side(min(width, height), sampled_fraction, self.cfg))
                    initial_crop_miss = bool(np.abs(original - crop[:2]).max() >= crop[2] / 2)
                    jitter = float(getattr(self.cfg, "OOF_CENTER_JITTER", 0.08))
                    if jitter < 0:
                        raise ValueError("OOF_CENTER_JITTER must be non-negative.")
                    crop[:2] += np.clip(np.random.normal(0, jitter, 2), -0.20, 0.20) * crop[2]
            else:
                sampled_fraction = float(np.random.choice(self.roi_fractions))
                side = float(research_roi_side(min(width, height), sampled_fraction, self.cfg))
                jitter = float(getattr(self.cfg, "TRAIN_CENTER_JITTER", 0.30))
                if jitter < 0:
                    raise ValueError("TRAIN_CENTER_JITTER must be non-negative.")
                # Truncated normal errors cover off-centre targets without
                # training on crops that cannot contain the annotated point.
                offset = np.clip(np.random.normal(0, jitter, 2), -0.42, 0.42) * side
                crop = np.r_[original + offset, side]
            sampled_side = float(crop[2])
            if self.training:
                # When optional prediction crops miss the point, increase the
                # *training* field of view; preserve the original coordinate.
                required_side = 2.5 * float(np.abs(original - crop[:2]).max()) + 2
                crop_was_enlarged = required_side > crop[2]
                crop[2] = max(crop[2], required_side)
            transform = research_crop_transform(*crop, self.size)
            image = cv2.warpAffine(image, transform[:2], (self.size, self.size),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        point = (transform @ np.r_[original, 1.0])[:2] if original is not None else np.zeros(2)
        if self.training:
            for _ in range(8):
                affine = cv2.getRotationMatrix2D(((self.size - 1) / 2,) * 2,
                                                np.random.uniform(-12, 12),
                                                np.random.uniform(0.95, 1.05))
                affine[:, 2] += np.random.uniform(-0.025, 0.025, 2) * self.size
                augmentation = np.vstack([affine, [0, 0, 1]])
                if np.random.random() < 0.5:
                    augmentation = np.array([[-1, 0, self.size - 1], [0, 1, 0], [0, 0, 1]]) @ augmentation
                candidate = (augmentation @ np.r_[point, 1.0])[:2]
                if np.all((candidate >= 0) & (candidate <= self.size - 1)):
                    image = cv2.warpAffine(image, augmentation[:2], (self.size, self.size),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
                    transform = augmentation @ transform
                    point = candidate
                    break
            image = np.clip(image.astype(np.float32) * np.random.uniform(0.85, 1.15)
                            + np.random.uniform(-8, 8), 0, 255).astype(np.uint8)
        xy = (point / (self.size - 1)).astype(np.float32)
        # Labels outside validation ROIs stay outside; no clipping or crop repair.
        # Validation uses original-pixel errors, not this possibly tiny heatmap.
        target = gaussian_heatmap(xy, self.hm_size, self.cfg.HEATMAP_SIGMA) if self.has_labels else np.zeros((1, self.hm_size, self.hm_size), np.float32)
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "target_heatmap": torch.from_numpy(target),
            "target_xy": torch.from_numpy(xy),
            "visible": torch.tensor(float(self.has_labels), dtype=torch.float32),
            "orig_wh": torch.tensor([width, height], dtype=torch.float32),
            "input_to_orig": torch.from_numpy(np.linalg.inv(transform)[:2].astype(np.float32)),
            "crop": torch.tensor(crop, dtype=torch.float32),
            "crop_from_prediction": torch.tensor(bool(crop_from_prediction), dtype=torch.bool),
            "crop_was_enlarged": torch.tensor(bool(crop_was_enlarged), dtype=torch.bool),
            "initial_crop_miss": torch.tensor(bool(initial_crop_miss), dtype=torch.bool),
            "crop_sampled_fraction": torch.tensor(sampled_fraction, dtype=torch.float32),
            "crop_sampled_side": torch.tensor(sampled_side if self.stage == "fine" else crop[2], dtype=torch.float32),
            "crop_final_side": torch.tensor(crop[2], dtype=torch.float32),
            "image_id": str(row.image_id),
        }
        if self.has_labels:
            item["x_true"] = torch.tensor(float(row.x), dtype=torch.float32)
            item["y_true"] = torch.tensor(float(row.y), dtype=torch.float32)
        return item


def make_research_loader(frame, cfg, training=False, stage="coarse", crop_spec=None, seed=0):
    return DataLoader(ResearchDataset(frame, cfg, training, stage, crop_spec),
                      batch_size=int(cfg.BATCH_SIZE), shuffle=training,
                      num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                      drop_last=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(int(seed)), persistent_workers=False)


def research_build_model(cfg, stage, device):
    """Require the explicitly resolved official weight file; no silent fallback."""
    path = getattr(cfg, "PRETRAINED_WEIGHTS_PATH", None)
    if cfg.PRETRAINED and (not path or not Path(path).is_file()):
        raise FileNotFoundError("Research training requires the attached official ResNet18 weight file. Resolve PRETRAINED_WEIGHTS_PATH first.")
    return FoveaNet(pretrained=bool(cfg.PRETRAINED), allow_download=False,
                    use_visibility=False, freeze_bn=bool(cfg.FREEZE_BN),
                    heatmap_size=int(cfg.HEATMAP_SIZE if stage == "coarse" else cfg.FINE_HEATMAP_SIZE),
                    pretrained_weights_path=path if cfg.PRETRAINED else None).to(device)


def research_schedule_factor(step, steps_per_epoch, horizon_epochs, warmup_epochs):
    """Epoch-normalized schedule: a shorter refit keeps the selection-run prefix.

    `horizon_epochs` is the configured stage budget, never the selected refit
    length. Different data sizes have different steps per epoch but the same
    epoch-level warmup/cosine trajectory.
    """
    if int(steps_per_epoch) < 1 or int(horizon_epochs) < 1 or int(step) < 0:
        raise ValueError("Schedule requires positive horizon/steps and nonnegative step.")
    if int(warmup_epochs) < 0:
        raise ValueError("WARMUP_EPOCHS must be nonnegative.")
    steps = int(horizon_epochs) * int(steps_per_epoch)
    warmup = min(int(warmup_epochs) * int(steps_per_epoch), max(0, steps - 1))
    if warmup and step < warmup:
        return (float(step) + 1.0) / warmup
    progress = min(1.0, max(0.0, (float(step) - warmup) / max(1, steps - warmup)))
    return 0.5 * (1 + math.cos(math.pi * progress))


def research_validation_loaders(validation, cfg, stage, val_crop, seed):
    """Prespecified fields of view on the supplied inner stopping split only.

    Centres are prediction-derived and stay fixed. Neither field-of-view
    selection nor resizing reads ground-truth coordinates.
    """
    if stage != "fine" or not bool(getattr(cfg, "FINE_MULTISCALE_EARLY_STOP", True)):
        return [("provided", make_research_loader(validation, cfg, False, stage, val_crop, seed))]
    if val_crop is None:
        raise ValueError("Multiscale fine early stopping requires prediction-derived centres.")
    crops = val_crop[["image_id", "crop_x", "crop_y", "crop_side"]].copy()
    crops["image_id"] = crops.image_id.astype(str)
    if crops.image_id.duplicated().any():
        raise ValueError("Validation crop IDs must be unique.")
    crops = crops.set_index("image_id").loc[validation.image_id.astype(str)].reset_index()
    short = validation[["width", "height"]].to_numpy(float).min(axis=1)
    fractions = tuple(float(x) for x in cfg.FINE_VALIDATION_ROI_FRACTIONS)
    if not fractions or len(set(fractions)) != len(fractions):
        raise ValueError("Fine validation scale list must be nonempty and unique.")
    loaders = []
    for fraction in fractions:
        spec = crops.copy()
        spec["crop_side"] = research_roi_side(short, fraction, cfg)
        loaders.append((f"fraction_{fraction:g}", make_research_loader(validation, cfg, False, "fine", spec, seed)))
    return loaders


def research_optimizer_groups(model, cfg):
    """Separate pretrained encoder/head rates, with explicit coverage checks."""
    multiplier = float(getattr(cfg, "BACKBONE_LR_MULTIPLIER", 1.0))
    base_lr = float(cfg.LR)
    if not math.isfinite(multiplier) or multiplier <= 0 or not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("Learning rates and backbone multiplier must be finite and positive.")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    encoder = [parameter for name, parameter in named if name.startswith("encoder.")]
    decoder = [parameter for name, parameter in named if not name.startswith("encoder.")]
    groups = [{"params": encoder, "lr": base_lr * multiplier, "name": "encoder"},
              {"params": decoder, "lr": base_lr, "name": "decoder"}]
    groups = [group for group in groups if group["params"]]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for _, parameter in named}
    if not ids or len(ids) != len(set(ids)) or set(ids) != expected:
        raise RuntimeError("Optimizer parameter groups omit or duplicate trainable parameters.")
    return groups


class ResearchEMA:
    """EMA copy with a short-run warm start; buffers always copy the online model.

    Only successful optimizer steps count. First update copies parameters;
    subsequent decay=min(max_decay,(1+updates)/(warmup+updates)). BN buffers
    remain synchronized, preserving the default frozen pretrained statistics.
    Validation and checkpoints use this same copy whenever EMA is enabled.
    """

    def __init__(self, model, decay=0.99, warmup_updates=10):
        self.max_decay = float(decay)
        self.warmup_updates = int(warmup_updates)
        if not math.isfinite(self.max_decay) or not 0 < self.max_decay < 1:
            raise ValueError("EMA_DECAY must lie strictly between zero and one.")
        if self.warmup_updates < 0 or self.warmup_updates != float(warmup_updates):
            raise ValueError("EMA_WARMUP_UPDATES must be a nonnegative integer.")
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.updates, self.last_decay = 0, 0.0

    @torch.no_grad()
    def update(self, model, optimizer_stepped=True):
        if not optimizer_stepped:
            return False
        self.updates += 1
        self.last_decay = (0.0 if self.updates == 1 else
                           min(self.max_decay, (1 + self.updates) / (self.warmup_updates + self.updates)))
        source_parameters = dict(model.named_parameters())
        target_parameters = dict(self.model.named_parameters())
        if source_parameters.keys() != target_parameters.keys():
            raise RuntimeError("EMA source and target parameter names differ.")
        for name, target in target_parameters.items():
            target.mul_(self.last_decay).add_(source_parameters[name].detach(), alpha=1 - self.last_decay)
        source_buffers = dict(model.named_buffers())
        target_buffers = dict(self.model.named_buffers())
        if source_buffers.keys() != target_buffers.keys():
            raise RuntimeError("EMA source and target buffer names differ.")
        for name, target in target_buffers.items():
            target.copy_(source_buffers[name].detach())
        return True


def fit_research_model(training, validation, cfg, device, stage, checkpoint_path,
                       train_crop=None, val_crop=None, epochs=None, seed=0):
    """Early-stop on inner data or refit for fixed epochs without validation.

    An integer epochs requests a final refit and the validation argument is
    intentionally never inspected. best_epoch is one-based, including refits.
    """
    device = torch.device(device)
    seed_everything(int(seed))
    fixed_epochs = epochs is not None
    horizon_epochs = int(cfg.COARSE_EPOCHS if stage == "coarse" else cfg.FINE_EPOCHS)
    total_epochs = int(epochs if fixed_epochs else horizon_epochs)
    minimum_epoch = int(getattr(cfg, "COARSE_MIN_EPOCHS" if stage == "coarse" else "FINE_MIN_EPOCHS", 1))
    if not 1 <= minimum_epoch <= horizon_epochs:
        raise ValueError("Minimum selectable epoch must lie within the training budget.")
    if total_epochs < 1 or len(training) == 0:
        raise ValueError("Training needs nonempty samples and at least one epoch.")
    if total_epochs > horizon_epochs:
        raise ValueError("Fixed refit epochs cannot exceed the configured stage schedule horizon.")
    train_loader = make_research_loader(training, cfg, True, stage, train_crop, seed)
    val_loaders = None
    if not fixed_epochs:
        if validation is None or len(validation) == 0:
            raise ValueError("Early stopping needs a nonempty inner validation set.")
        if "group" in training and "group" in validation:
            if set(training.group.astype(str)) & set(validation.group.astype(str)):
                raise ValueError("Training/validation groups overlap.")
        val_loaders = research_validation_loaders(validation, cfg, stage, val_crop, seed)
    model = research_build_model(cfg, stage, device)
    optimizer = torch.optim.AdamW(research_optimizer_groups(model, cfg),
                                  lr=float(cfg.LR), weight_decay=float(cfg.WEIGHT_DECAY))
    group_metadata = [{"name": group["name"], "base_lr": group["lr"],
                       "parameter_tensors": len(group["params"]),
                       "parameters": sum(parameter.numel() for parameter in group["params"])}
                      for group in optimizer.param_groups]
    use_ema = bool(getattr(cfg, "USE_EMA", False))
    ema = ResearchEMA(model, cfg.EMA_DECAY, cfg.EMA_WARMUP_UPDATES) if use_ema else None
    total_successful_steps = 0
    def factor(step):
        return research_schedule_factor(step, len(train_loader), horizon_epochs, cfg.WARMUP_EPOCHS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    amp = bool(cfg.AMP) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=float(getattr(cfg, "AMP_INIT_SCALE", 1024.0)))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=float(getattr(cfg, "AMP_INIT_SCALE", 1024.0)))
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, bad_epochs = [], math.inf, 0, 0
    for epoch in range(1, total_epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss, count = 0.0, 0
        crop_counts = {"crop_from_prediction": 0, "crop_was_enlarged": 0, "initial_crop_miss": 0}
        sampled_side_sum, final_side_sum, scale_counts = 0.0, 0.0, {}
        component_sums = {"heatmap": 0.0, "coordinate": 0.0, "visibility": 0.0}
        successful_steps = 0
        for cpu_batch in train_loader:
            batch = _to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, cfg):
                output = model(batch["image"])
            loss, components = fovea_loss(output, batch, cfg, use_visibility=False)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite {stage} loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.GRAD_CLIP))
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_stepped = scaler.get_scale() >= previous_scale
            if optimizer_stepped:
                scheduler.step()
                successful_steps += 1
                total_successful_steps += 1
            if ema is not None:
                ema.update(model, optimizer_stepped=optimizer_stepped)
            n = int(batch["image"].shape[0])
            total_loss += float(loss.detach()) * n
            for component, value in components.items():
                component_sums[component] += value * n
            count += n
            for key in crop_counts:
                crop_counts[key] += int(batch[key].sum())
            sampled_side_sum += float(batch["crop_sampled_side"].sum())
            final_side_sum += float(batch["crop_final_side"].sum())
            for fraction in batch["crop_sampled_fraction"].detach().cpu().tolist():
                if fraction > 0:
                    key = f"{fraction:.4f}"
                    scale_counts[key] = scale_counts.get(key, 0) + 1
        evaluation_model = ema.model if ema is not None else model
        if fixed_epochs:
            score, scale_scores = None, {}
        else:
            score, scale_scores = optimized_validation_score(evaluation_model, val_loaders, cfg, device, stage)
        if score is not None and not math.isfinite(score):
            raise FloatingPointError("Non-finite inner validation RMSE.")
        eligible = fixed_epochs or epoch >= minimum_epoch
        improved = fixed_epochs or (eligible and score < best_score)
        if improved:
            best_epoch, bad_epochs = epoch, 0
            if score is not None:
                best_score = float(score)
            if not fixed_epochs or epoch == total_epochs:
                torch.save({"model": {k: v.detach().cpu() for k, v in evaluation_model.state_dict().items()},
                            "epoch": epoch, "stage": stage, "use_visibility": False,
                            "heatmap_size": model.heatmap_size,
                            "schedule_horizon_epochs": horizon_epochs,
                            "validation_tta": bool(cfg.TTA),
                            "validation_scale_scores": scale_scores,
                            "weights_kind": "ema" if use_ema else "online",
                            "ema_updates": ema.updates if ema is not None else 0,
                            "optimizer_groups": group_metadata,
                            "selection": "fixed_epochs_no_validation" if fixed_epochs else "inner_validation_rmse",
                            "rmse": score, "seed": int(seed)}, checkpoint_path)
        elif eligible:
            bad_epochs += 1
        record = {"stage": stage, "epoch": epoch, "train_loss": total_loss / count,
                  "val_rmse": score, "seconds": time.perf_counter() - started,
                  "learning_rate": optimizer.param_groups[0]["lr"],
                  "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                  "train_heatmap_kl": component_sums["heatmap"] / count,
                  "train_coordinate_mse": component_sums["coordinate"] / count,
                  "train_visibility_bce": component_sums["visibility"] / count,
                  "successful_optimizer_steps": successful_steps,
                  "ema_updates": ema.updates if ema is not None else 0,
                  "ema_effective_decay": ema.last_decay if ema is not None else None,
                  "schedule_horizon_epochs": horizon_epochs, **crop_counts,
                  "crop_sampled_side_mean_px": sampled_side_sum / count,
                  "crop_final_side_mean_px": final_side_sum / count,
                  "crop_enlargement_rate": crop_counts["crop_was_enlarged"] / count,
                  "crop_sampled_fraction_counts": scale_counts,
                  "validation_scale_scores": scale_scores}
        history.append(record)
        validation_message = "fixed-epoch refit" if fixed_epochs else f"inner RMSE {score:.3f}px"
        print(f"{stage} | {checkpoint_path.stem} | epoch {epoch}/{total_epochs} | "
              f"loss {record['train_loss']:.4f} | {validation_message} | {record['seconds']:.1f}s", flush=True)
        if not fixed_epochs and int(cfg.PATIENCE) > 0 and bad_epochs >= int(cfg.PATIENCE):
            break
    load_model_checkpoint(model, checkpoint_path, device)
    info = {"best_epoch": best_epoch, "best_rmse": None if fixed_epochs else best_score,
            "history": history, "checkpoint_path": str(checkpoint_path),
            "fixed_epochs": fixed_epochs, "training_images": len(training),
            "schedule_horizon_epochs": horizon_epochs, "requested_training_epochs": total_epochs,
            "validation_tta": bool(cfg.TTA),
            "optimizer_groups": group_metadata,
            "weights_kind": "ema" if use_ema else "online",
            "successful_optimizer_steps": total_successful_steps,
            "ema_updates": ema.updates if ema is not None else 0,
            "ema_max_decay": ema.max_decay if ema is not None else None,
            "validation_metric": ("fixed 0.5 coarse + 0.5 multiscale fine coordinate RMSE" if stage == "fine" else "coarse TTA coordinate RMSE"),
            "minimum_selectable_epoch": minimum_epoch,
            "fine_multiscale_train": bool(getattr(cfg, "FINE_MULTISCALE_TRAIN", True)),
            "crop_training_policy": ("prediction centre + optional multiscale side + jitter; enlarge around that centre when training target misses; never change labels"
                                     if stage == "fine" and train_crop is not None else
                                     "GT centre + truncated jitter" if stage == "fine" else "full image")}
    checkpoint_path.with_suffix(".history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return model, info


def research_distribution_moments(logits, input_to_orig, input_size, undo_hflip=False):
    """Probability mean/covariance in original pixels, with affine propagation."""
    logits = logits.float()
    if undo_hflip:
        logits = torch.flip(logits, dims=[-1])
    b, _, h, w = logits.shape
    prob = logits.flatten(1).softmax(1)
    y, x = torch.meshgrid(torch.linspace(0, input_size - 1, h, device=logits.device),
                          torch.linspace(0, input_size - 1, w, device=logits.device), indexing="ij")
    grid = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    mean = prob @ grid
    delta = grid.unsqueeze(0) - mean.unsqueeze(1)
    covariance = torch.einsum("bn,bni,bnj->bij", prob, delta, delta)
    linear, shift = input_to_orig.float()[:, :, :2], input_to_orig.float()[:, :, 2]
    original_mean = torch.bmm(linear, mean.unsqueeze(-1)).squeeze(-1) + shift
    original_covariance = linear @ covariance @ linear.transpose(1, 2)
    return original_mean, original_covariance


@torch.inference_mode()
def research_predict_batch(model, batch, cfg, device):
    """Shared original-coordinate mean and total covariance for train selection/inference."""
    with _autocast_context(device, cfg):
        output = model(batch["image"])
    size = int(batch["image"].shape[-1])
    mean, covariance = research_distribution_moments(output["logits"], batch["input_to_orig"], size)
    means, covariances = [mean], [covariance]
    if cfg.TTA:
        with _autocast_context(device, cfg):
            flipped = model(torch.flip(batch["image"], dims=[-1]))
        mean, covariance = research_distribution_moments(flipped["logits"], batch["input_to_orig"], size, True)
        means.append(mean)
        covariances.append(covariance)
    stacked = torch.stack(means)
    mean = stacked.mean(0)
    offsets = stacked - mean.unsqueeze(0)
    covariance = (torch.stack(covariances) + offsets.unsqueeze(-1) @ offsets.unsqueeze(-2)).mean(0)
    covariance = (covariance + covariance.transpose(1, 2)) / 2
    mean = torch.minimum(mean.clamp_min(0), batch["orig_wh"].float() - 1)
    return mean, covariance


@torch.inference_mode()
def predict_research(model, frame, cfg, device, stage="coarse", crop_spec=None):
    """No label access. Mixture covariance includes disagreement between TTA views.

    uncertainty = sqrt(largest eigenvalue of original-pixel covariance), in px.
    inference_seconds includes data loading and inference, averaged per image.
    Crop-edge fraction is min distance to the crop's four pixel edges / side;
    it may be negative if final image-boundary clipping moves a point out of ROI.
    """
    device = torch.device(device)
    # An explicit whitelist prevents accidental use of x/y/visible at inference.
    inference_frame = frame[["image_id", "path", "width", "height"]].copy()
    loader = make_research_loader(inference_frame, cfg, False, stage, crop_spec, int(cfg.SEED))
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started, rows = time.perf_counter(), []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        mean, covariance = research_predict_batch(model, batch, cfg, device)
        uncertainty = torch.linalg.eigvalsh(covariance).clamp_min(0)[:, -1].sqrt()
        crop = batch["crop"]
        edge = 0.5 - ((mean - crop[:, :2]).abs() / crop[:, 2:3]).amax(1)
        for i, image_id in enumerate(batch["image_id"]):
            rows.append({"image_id": str(image_id), "x": float(mean[i, 0]), "y": float(mean[i, 1]),
                         "uncertainty": float(uncertainty[i]), "cov_xx": float(covariance[i, 0, 0]),
                         "cov_xy": float(covariance[i, 0, 1]), "cov_yy": float(covariance[i, 1, 1]),
                         "crop_edge_fraction": float(edge[i]) if stage == "fine" else np.nan})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    columns = ["image_id", "x", "y", "uncertainty", "cov_xx", "cov_xy", "cov_yy", "crop_edge_fraction"]
    result = pd.DataFrame(rows, columns=columns)
    result["inference_seconds"] = (time.perf_counter() - started) / max(1, len(result))
    return result


def _research_setting(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def _research_frame(frame, columns, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = set(["image_id", *columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not len(frame):
        raise ValueError(f"{name} is empty")
    out = frame.copy()
    if out.image_id.isna().any():
        raise ValueError(f"{name} has null image_id")
    out["image_id"] = out.image_id.astype(str)
    if out.image_id.duplicated().any():
        raise ValueError(f"{name} has duplicate image_id")
    for col in columns:
        if col == "group":
            if out[col].isna().any():
                raise ValueError(f"{name} has null groups")
            out[col] = out[col].astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="raise")
            if not np.isfinite(out[col].to_numpy(dtype=float)).all():
                raise ValueError(f"{name}.{col} contains NaN or infinity")
    return out


def _research_align(pred, truth, pred_columns=("x", "y"), truth_columns=("x", "y")):
    pred = _research_frame(pred, pred_columns, "predictions")
    truth = _research_frame(truth, truth_columns, "reference frame")
    if set(pred.image_id) != set(truth.image_id):
        raise ValueError("Prediction/reference image_id sets differ")
    truth = truth.set_index("image_id").loc[pred.image_id].reset_index()
    return pred.reset_index(drop=True), truth.reset_index(drop=True)


def _research_uncertainty(pred, floor_px):
    floor_px = float(floor_px)
    if not math.isfinite(floor_px) or floor_px <= 0:
        raise ValueError("UNCERTAINTY_FLOOR_PX must be finite and positive")
    value = pred.uncertainty.to_numpy(dtype=float)
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("uncertainty must be finite and nonnegative")
    return np.maximum(value, floor_px)


def make_roi_spec(pred, frame, cfg, mode="fixed", calibration=None, fixed_fraction=None):
    """Create centered square ROIs using predictions and dimensions only.

    crop_x/crop_y are CENTER coordinates in original pixels. Crops extending
    beyond the image use padding; effective image intersections are recorded.
    All modes apply side=clip(base_side+2*margin,min_fraction*short,max_fraction*short).
    A fixed fraction describes base_side before this common margin and clamp.
    """
    if mode not in {"fixed", "raw", "calibrated"}:
        raise ValueError(f"Unsupported ROI mode: {mode}")
    required = ("x", "y") if mode == "fixed" else ("x", "y", "uncertainty")
    pred, frame = _research_align(pred, frame, required, ("width", "height"))
    dimensions = frame[["width", "height"]].to_numpy(float)
    if np.any(dimensions <= 1):
        raise ValueError("Image dimensions must exceed one pixel")
    short = dimensions.min(axis=1)
    min_fraction = float(_research_setting(cfg, "ROI_MIN_FRACTION", 0.15))
    max_fraction = float(_research_setting(cfg, "ROI_MAX_FRACTION", 0.60))
    margin = float(_research_setting(cfg, "ROI_MARGIN_PX", 16.0))
    if not 0 < min_fraction <= max_fraction <= 1 or not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid ROI fraction bounds or pixel margin")
    floor_px = float(_research_setting(cfg, "UNCERTAINTY_FLOOR_PX", 2.0))
    if mode == "fixed":
        fraction = float(fixed_fraction if fixed_fraction is not None else
                         _research_setting(cfg, "FIXED_ROI_FRACTION", 0.30))
        if not math.isfinite(fraction) or fraction <= 0:
            raise ValueError("Fixed ROI fraction must be finite and positive")
        base_side = fraction * short
    else:
        if mode == "calibrated":
            if calibration is None or "alpha" not in calibration:
                raise ValueError("Calibrated ROIs require fitted alpha")
            multiplier = float(calibration["alpha"])
            floor_px = float(calibration.get("uncertainty_floor_px", floor_px))
        else:
            multiplier = float(_research_setting(cfg, "RAW_RADIUS_MULTIPLIER", 2.0))
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("Radius multiplier must be finite and nonnegative")
        base_side = 2 * multiplier * _research_uncertainty(pred, floor_px)
    proposed = base_side + 2 * margin
    if not np.isfinite(proposed).all():
        raise ValueError("Proposed ROI sides overflowed or contain non-finite values")
    side = np.clip(proposed, min_fraction * short, max_fraction * short)
    center = pred[["x", "y"]].to_numpy(float)
    if np.any(center < 0) or np.any(center > dimensions - 1 + 1e-6):
        raise ValueError("ROI centers must be within their original image")
    lower = center - side[:, None] / 2
    upper = center + side[:, None] / 2
    effective_lower = np.maximum(lower, 0)
    effective_upper = np.minimum(upper, dimensions - 1)
    # Areas are geometric fields in pixel-center coordinates; do not add +1.
    # Geometric crop area remains side**2 for matching the resampling field.
    effective_area = np.maximum(effective_upper[:, 0] - effective_lower[:, 0], 0) * np.maximum(effective_upper[:, 1] - effective_lower[:, 1], 0)
    return pd.DataFrame({
        "image_id": pred.image_id,
        "crop_x": center[:, 0], "crop_y": center[:, 1], "crop_side": side,
        "roi_x0": effective_lower[:, 0], "roi_y0": effective_lower[:, 1],
        "roi_x1": effective_upper[:, 0], "roi_y1": effective_upper[:, 1],
        "roi_area_px2": side ** 2, "roi_effective_area_px2": effective_area,
        "roi_area_fraction": side ** 2 / np.prod(dimensions, axis=1),
        "roi_at_min": proposed < min_fraction * short,
        "roi_at_max": proposed > max_fraction * short,
        "roi_has_padding": np.any(lower < 0, axis=1) | np.any(upper > dimensions - 1, axis=1),
        "roi_mode": mode,
    })


def roi_contains_truth(roi, truth):
    """Evaluation only. Never call this to select per-image inference behavior."""
    roi, truth = _research_align(roi, truth, ("roi_x0", "roi_y0", "roi_x1", "roi_y1"))
    point = truth[["x", "y"]].to_numpy(float)
    return pd.Series(
        (point[:, 0] >= roi.roi_x0.to_numpy(float)) & (point[:, 0] <= roi.roi_x1.to_numpy(float)) &
        (point[:, 1] >= roi.roi_y0.to_numpy(float)) & (point[:, 1] <= roi.roi_y1.to_numpy(float)),
        index=roi.image_id, name="contains_truth",
    )


def localization_metrics(pred, truth, roi=None):
    pred, truth = _research_align(pred, truth)
    delta = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
    distance = np.linalg.norm(delta, axis=1)
    result = {
        "n_images": int(len(pred)), "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mean_distance_px": float(distance.mean()),
        "median_distance_px": float(np.median(distance)),
        "p90_distance_px": float(np.quantile(distance, 0.90)),
        "max_distance_px": float(distance.max()),
    }
    if "group" in truth:
        result["n_groups"] = int(truth.group.nunique())
    if "fallback_used" in pred:
        result["fallback_rate"] = float(pred.fallback_used.astype(bool).mean())
    if roi is not None:
        roi, truth_roi = _research_align(roi, truth, ("crop_side", "roi_x0", "roi_y0", "roi_x1", "roi_y1"))
        result["roi_coverage"] = float(roi_contains_truth(roi, truth_roi).mean())
        result["mean_roi_area_px2"] = float(np.mean(roi.crop_side.to_numpy(float) ** 2))
        for column, key in [("roi_area_fraction", "mean_roi_area_fraction"),
                            ("roi_effective_area_px2", "mean_roi_effective_area_px2"),
                            ("roi_at_max", "roi_max_clamp_rate"),
                            ("roi_at_min", "roi_min_clamp_rate"),
                            ("roi_has_padding", "roi_padding_rate")]:
            if column in roi:
                value = pd.to_numeric(roi[column], errors="raise").to_numpy(float)
                if not np.isfinite(value).all():
                    raise ValueError(f"ROI metric {column} contains non-finite values")
                result[key] = float(value.mean())
    return result


def paired_group_bootstrap(predictions_by_method, truth, baseline=None, n_bootstrap=2000, seed=2026):
    """Paired percentile intervals conditional on these fitted OOF predictions.

    Resample whole groups; preserve all labeled rows within each sampled group.
    This does not account for retraining variation or unknown patient overlap.
    """
    if not predictions_by_method:
        raise ValueError("At least one prediction method is required")
    names = list(predictions_by_method)
    baseline = names[0] if baseline is None else baseline
    if baseline not in names or int(n_bootstrap) < 100:
        raise ValueError("Invalid bootstrap baseline or fewer than 100 resamples")
    truth = _research_frame(truth, ("x", "y", "group"), "bootstrap truth")
    # Establish one reference order for all methods, regardless of their row order.
    reference_ids = truth.image_id.tolist()
    group_codes, group_names = pd.factorize(truth.group, sort=True)
    n_groups = len(group_names)
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    sums_by_method = []
    point_rmse = []
    for name in names:
        pred, aligned_truth = _research_align(predictions_by_method[name], truth)
        pred = pred.set_index("image_id").loc[reference_ids]
        error = pred[["x", "y"]].to_numpy(float) - truth[["x", "y"]].to_numpy(float)
        squared = np.mean(error ** 2, axis=1)
        sums_by_method.append(np.bincount(group_codes, weights=squared, minlength=n_groups))
        point_rmse.append(float(np.sqrt(squared.mean())))
    rng = np.random.default_rng(int(seed))
    multiplicity = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=int(n_bootstrap))
    denominator = multiplicity @ counts
    scores = np.sqrt((multiplicity @ np.asarray(sums_by_method).T) / denominator[:, None])
    baseline_index = names.index(baseline)
    records = []
    for index, name in enumerate(names):
        delta = scores[:, index] - scores[:, baseline_index]
        records.append({"method": name, "baseline": baseline, "rmse": point_rmse[index],
                        "rmse_ci_low": float(np.quantile(scores[:, index], 0.025)),
                        "rmse_ci_high": float(np.quantile(scores[:, index], 0.975)),
                        "delta_rmse": point_rmse[index] - point_rmse[baseline_index],
                        "delta_ci_low": float(np.quantile(delta, 0.025)),
                        "delta_ci_high": float(np.quantile(delta, 0.975)),
                        "n_images": int(len(truth)), "n_groups": int(n_groups),
                        "n_bootstrap": int(n_bootstrap), "seed": int(seed),
                        "ci_note": "Paired group percentile bootstrap of fixed OOF predictions; excludes retraining variation and unobserved patient dependence."})
    return pd.DataFrame(records)


PIPELINE_VERSION = "2026task3-fovea-1.2-real-data"


def json_save(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def apply_visibility(frame, threshold, coordinate_origin=0):
    prediction = frame[["x", "y"]].to_numpy(dtype=np.float64).copy()
    keep = frame.visibility_prob.to_numpy(float) >= threshold
    prediction[keep] += coordinate_origin
    prediction[~keep] = 0.0
    return prediction


def truth_in_submission_coordinates(frame, coordinate_origin=0):
    truth = frame[["x_true", "y_true"]].to_numpy(dtype=np.float64).copy()
    truth[frame.visible_true.to_numpy(bool)] += coordinate_origin
    return truth


def write_submission(sample, layout, prediction_frame, metadata, cfg, threshold):
    if prediction_frame.image_id.duplicated().any():
        raise ValueError("Duplicate predictions for a test image.")
    if set(prediction_frame.image_id) != set(layout.image_id):
        raise ValueError("Predicted image IDs do not match the sample submission.")
    prediction = apply_visibility(prediction_frame, threshold, cfg.COORDINATE_ORIGIN)
    mapping = {key: point for key, point in zip(prediction_frame.image_id, prediction)}
    values = [mapping[row.image_id][0 if row.axis == "x" else 1] for row in layout.itertuples()]
    result = sample.copy()
    result[metadata["value_column"]] = np.asarray(values, dtype=np.float64)
    assert len(result) == len(sample)
    assert result[metadata["id_column"]].equals(sample[metadata["id_column"]])
    if not np.isfinite(result[metadata["value_column"]]).all():
        raise ValueError("Submission contains non-finite values.")
    for row, point in zip(prediction_frame.itertuples(), prediction):
        if np.all(point == 0):
            continue
        low = cfg.COORDINATE_ORIGIN
        upper = np.array([row.width - 1, row.height - 1]) + low
        if np.any(point < low - 1e-6) or np.any(point > upper + 1e-6):
            raise ValueError(f"Out-of-bounds prediction: {row.image_id} -> {point}")
    path = Path(cfg.SUBMISSION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.6f")
    # Round-trip through disk: verify actual exported IDs and numeric values.
    reread = pd.read_csv(path, dtype={metadata["id_column"]: str}, keep_default_na=False)
    assert reread[metadata["id_column"]].tolist() == sample[metadata["id_column"]].tolist()
    assert np.isfinite(pd.to_numeric(reread[metadata["value_column"]], errors="raise")).all()
    print(f"Submission saved: {path} ({len(result)} rows)")
    print(result.head(8).to_string(index=False))
    return result


def plot_oof_errors(oof, train, threshold, cfg, output_dir):
    points = apply_visibility(oof, threshold, coordinate_origin=0)
    scored_points = apply_visibility(oof, threshold, cfg.COORDINATE_ORIGIN)
    scored_truth = truth_in_submission_coordinates(oof, cfg.COORDINATE_ORIGIN)
    errors = np.linalg.norm(scored_points - scored_truth, axis=1)
    report = oof.copy()
    report["pred_x"] = points[:, 0]
    report["pred_y"] = points[:, 1]
    report["euclidean_error_px"] = errors
    report = report.merge(train[["image_id", "path"]], on="image_id", validate="one_to_one")
    report.to_csv(Path(output_dir) / "oof_error_details.csv", index=False)
    worst = report.nlargest(min(12, len(report)), "euclidean_error_px")
    fig, axes = plt.subplots(math.ceil(len(worst) / 4), 4, figsize=(16, 4 * math.ceil(len(worst) / 4)), squeeze=False)
    for axis, row in zip(axes.flat, worst.itertuples()):
        axis.imshow(read_rgb(row.path))
        if row.visible_true:
            axis.scatter(row.x_true, row.y_true, marker="+", color="lime", s=120, label="truth")
        if row.visibility_prob >= threshold:
            axis.scatter(row.pred_x, row.pred_y, marker="x", color="red", s=90, label="prediction")
        axis.set_title(f"{row.image_id} | error {row.euclidean_error_px:.1f}px\np_visible={row.visibility_prob:.2f}", fontsize=9)
        axis.axis("off")
    for axis in list(axes.flat)[len(worst):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "oof_worst_cases.png", dpi=120, bbox_inches="tight")
    if cfg.SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return report


def _research_cleanup(model=None):
    if model is not None:
        model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def research_group_holdout(frame, fraction, seed):
    groups = frame.group.unique().copy()
    if len(groups) < 5:
        raise ValueError("Too few groups for a development/calibration split.")
    np.random.default_rng(seed).shuffle(groups)
    count = min(len(groups) - 3, max(2, int(round(len(groups) * fraction))))
    selected = set(groups[:count])
    fit = frame.loc[~frame.group.isin(selected)].reset_index(drop=True)
    held = frame.loc[frame.group.isin(selected)].reset_index(drop=True)
    assert not set(fit.group) & set(held.group)
    return fit, held


def research_file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def research_runtime_code_digest():
    """Detect edited notebook function bodies without relying on source files.

    Cell execution filenames/line offsets are normalized so Run All can reuse
    the same code. Nested code objects are normalized recursively as well.
    """
    import types
    import inspect
    def normalize(value):
        # marshal encodes object sharing/interning flags which can change after
        # imports or training. Hash semantic values instead of those flags.
        if isinstance(value, types.CodeType):
            return {"bytecode": value.co_code.hex(), "constants": normalize(value.co_consts),
                    "names": value.co_names, "variables": value.co_varnames,
                    "free": value.co_freevars, "cells": value.co_cellvars,
                    "arguments": [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
                    "flags": value.co_flags}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, frozenset):
            return {"frozenset": sorted([normalize(item) for item in value], key=lambda item: json.dumps(item, sort_keys=True))}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, (float, complex)):
            return {type(value).__name__: repr(value)}
        if value is Ellipsis:
            return {"ellipsis": True}
        return value
    digest = hashlib.sha256()
    for name, value in sorted(list(globals().items())):
        entries = []
        if isinstance(value, types.FunctionType):
            value = inspect.unwrap(value)
        if isinstance(value, types.FunctionType) and value.__globals__ is globals():
            entries.append((name, value))
        elif isinstance(value, type) and value.__module__ == __name__:
            for member, function in sorted(vars(value).items()):
                if isinstance(function, (staticmethod, classmethod)):
                    function = function.__func__
                if isinstance(function, types.FunctionType):
                    function = inspect.unwrap(function)
                    entries.append((name + "." + member, function))
        for label, function in entries:
            digest.update(label.encode())
            digest.update(json.dumps(normalize(function.__code__), sort_keys=True, ensure_ascii=True).encode())
            digest.update(json.dumps(normalize(function.__defaults__), sort_keys=True, default=str).encode())
            digest.update(json.dumps(function.__kwdefaults__, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def research_training_environment():
    import importlib.metadata
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": importlib.metadata.version("torchvision"),
            "numpy": np.__version__, "pandas": pd.__version__, "opencv": cv2.__version__,
            "device_type": "cuda" if torch.cuda.is_available() else "cpu"}


def research_cache_manifest(folder, paths):
    folder = Path(folder).resolve()
    result = {}
    for path in sorted(map(Path, paths)):
        resolved = path.resolve()
        relative = resolved.relative_to(folder).as_posix()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[relative] = {"bytes": resolved.stat().st_size, "sha256": research_file_digest(resolved)}
    if not result:
        raise ValueError("Cannot mark an empty cache as complete.")
    return result


def research_verify_cache(folder, manifest, required_paths=()):
    folder = Path(folder).resolve()
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{folder}: cache has no integrity manifest; use a new OUTPUT_DIR or REUSE_CHECKPOINTS=False.")
    missing = set(required_paths) - set(manifest)
    if missing:
        raise ValueError(f"Cache integrity manifest is missing required entries: {sorted(missing)}")
    for relative, expected in manifest.items():
        path = (folder / relative).resolve()
        try:
            path.relative_to(folder)
        except ValueError as exc:
            raise ValueError("Cache manifest escapes its output directory") from exc
        if (not path.is_file() or path.stat().st_size != expected.get("bytes") or
                research_file_digest(path) != expected.get("sha256")):
            raise ValueError(f"Cache integrity check failed: {path}. Restore the file or set REUSE_CHECKPOINTS=False.")


def research_stage_signature(training, validation, cfg, stage, seed, epochs, train_crop, val_crop):
    def frame_payload(frame, crop=False):
        if frame is None:
            return None
        names = (["image_id", "crop_x", "crop_y", "crop_side"] if crop else
                 ["image_id", "pixel_hash", "x", "y", "visible", "group", "width", "height"])
        cols = [name for name in names if name in frame]
        return frame[cols].to_dict("records")
    config = asdict(cfg)
    # Worker count affects augmentation RNG. Keep it in the identity.
    for name in ("OUTPUT_DIR", "SUBMISSION_PATH", "REUSE_CHECKPOINTS", "SHOW_PLOTS",
                 "PRETRAINED_WEIGHTS_PATH", "INPUT_ROOT", "DATA_DIR", "LABEL_CSV", "SAMPLE_CSV"):
        config.pop(name, None)
    payload = {"source": globals().get("RESEARCH_SOURCE_HASH", RESEARCH_VERSION),
               "runtime_code": research_runtime_code_digest(),
               "environment": research_training_environment(), "config": config, "stage": stage,
               "seed": int(seed), "epochs": epochs, "training": frame_payload(training),
               "validation": frame_payload(validation) if epochs is None else None,
               "train_crop": frame_payload(train_crop, True),
               "val_crop": frame_payload(val_crop, True) if epochs is None else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _research_stage(training, validation, cfg, device, stage, folder, seed,
                    train_crop=None, val_crop=None, epochs=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "stage_complete.json"
    signature = research_stage_signature(training, validation, cfg, stage, seed, epochs, train_crop, val_crop)
    if cfg.REUSE_CHECKPOINTS and marker.exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record.get("signature") != signature:
            raise ValueError(f"{folder}: cached stage configuration/data/code changed; use a new OUTPUT_DIR or REUSE_CHECKPOINTS=False.")
        research_verify_cache(folder, record.get("files"), ("model.pt", "history.csv", "training_summary.json"))
        model = research_build_model(cfg, stage, device)
        load_model_checkpoint(model, folder / "model.pt", device)
        info = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
        info["history"] = pd.read_csv(folder / "history.csv").to_dict("records")
        info["reused_completed_stage"] = True
        print(f"Reuse completed training stage: {folder.name}", flush=True)
        return model, info
    if marker.exists():
        marker.unlink()
    model, info = fit_research_model(
        training, validation, cfg, device, stage, folder / "model.pt",
        train_crop=train_crop, val_crop=val_crop, epochs=epochs, seed=seed)
    pd.DataFrame(info["history"]).to_csv(folder / "history.csv", index=False)
    safe_info = {k: v for k, v in info.items() if k != "history"}
    json_save(safe_info, folder / "training_summary.json")
    files = [folder / name for name in ("model.pt", "history.csv", "training_summary.json")]
    record = {"signature": signature, "files": research_cache_manifest(folder, files),
              "resume_policy": "Reuse completed stages only; interrupted epochs restart this stage."}
    temporary = marker.with_suffix(".partial.json")
    json_save(record, temporary)
    temporary.replace(marker)
    return model, info


def _research_attach_truth(pred, truth, fold, experiment):
    cols = ["image_id", "x", "y", "group", "width", "height"]
    labels = truth[cols].rename(columns={"x": "x_true", "y": "y_true"})
    base = pred.drop(columns=[c for c in ("group", "width", "height", "x_true", "y_true")
                              if c in pred], errors="ignore")
    out = base.merge(labels, on="image_id", validate="one_to_one")
    out["fold"], out["experiment"] = fold, experiment
    out["error_px"] = np.hypot(out.x - out.x_true, out.y - out.y_true)
    return out


RESEARCH_SOURCE_HASH = "ff7cda2ef1fa2bb551e7f72eee3dc24d254db255601947b2ae497fc170cf4563"


@dataclass
class OptimizedConfig(ResearchConfig):
    AMP_INIT_SCALE: float = 1024.0
    OUTPUT_DIR: str = "/kaggle/working/optimized_20261026"
    SUBMISSION_PATH: str = "/kaggle/working/submission.csv"
    INPUT_SIZE: int = 384
    HEATMAP_SIZE: int = 96
    FINE_INPUT_SIZE: int = 384
    FINE_HEATMAP_SIZE: int = 96
    COARSE_EPOCHS: int = 35
    FINE_EPOCHS: int = 30
    COARSE_MIN_EPOCHS: int = 10
    FINE_MIN_EPOCHS: int = 8
    PATIENCE: int = 10
    BACKBONE_LR_MULTIPLIER: float = 0.25
    ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)
    FINE_VALIDATION_ROI_FRACTIONS: tuple = (0.20, 0.30, 0.40)
    FINE_MULTISCALE_TRAIN: bool = True
    FINE_MULTISCALE_EARLY_STOP: bool = True
    SELECT_FINE_CROP_POLICY: bool = False
    COARSE_BLEND_WEIGHT: float = 0.5
    OOF_CENTER_JITTER: float = 0.10
    NUM_WORKERS: int = 0
    BATCH_SIZE: int = 8
    AMP: bool = True
    USE_EMA: bool = False
    EARLY_STOP_FRACTION: float = 0.20
    REUSE_CHECKPOINTS: bool = True
    SHOW_PLOTS: bool = False
    SMOKE_TEST: bool = False
    N_FOLDS: int = 5


RESEARCH_VERSION = "4.0.0-20261026"


RESEARCH_SOURCE_HASH = "20261026-full-outer-fit-multiscale-blend-v1"


@torch.inference_mode()
def optimized_validation_score(model, loaders, cfg, device, stage):
    """Evaluate exactly the deployed TTA + multiscale + fixed blend on inner data."""
    model.eval()
    predictions, reference_truth, reference_ids = [], None, None
    scale_scores = {}
    for name, loader in loaders:
        pp, tt, ids = [], [], []
        for cpu_batch in loader:
            batch = _to_device(cpu_batch, device)
            point, _ = research_predict_batch(model, batch, cfg, device)
            pp.append(point.double().cpu().numpy())
            tt.append(torch.stack([batch["x_true"], batch["y_true"]], dim=1).double().cpu().numpy())
            ids.extend(list(cpu_batch['image_id']))
        prediction, truth = np.concatenate(pp), np.concatenate(tt)
        if reference_ids is None:
            reference_ids, reference_truth = ids, truth
        else:
            if ids != reference_ids or not np.array_equal(reference_truth, truth):
                raise AssertionError("Multiscale validation rows must be aligned.")
        predictions.append(prediction)
        scale_scores[name] = float(np.sqrt(np.mean((prediction-truth)**2)))
    mean = np.mean(predictions, axis=0)
    if stage == 'fine':
        dataset = loaders[0][1].dataset
        coarse = dataset.crops.loc[reference_ids, ['crop_x', 'crop_y']].to_numpy(float)
        weight = float(cfg.COARSE_BLEND_WEIGHT)
        if not 0 <= weight <= 1:
            raise ValueError("Invalid fixed blend weight.")
        mean = weight*coarse + (1-weight)*mean
    score = float(np.sqrt(np.mean((mean-reference_truth)**2)))
    scale_scores['deployed_combination'] = score
    return score, scale_scores


def optimized_stage(training, validation, cfg, device, stage, folder, seed,
                    train_crop=None, val_crop=None, epochs=None):
    """Persist the exact IDs actually passed to each training/selection call."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if validation is not None and epochs is None:
        assert not set(training.group)&set(validation.group)
    columns=['image_id','group','path','pixel_hash']
    training[columns].to_csv(folder/'actual_training_inputs.csv',index=False)
    if validation is not None and epochs is None:
        validation[columns].to_csv(folder/'actual_early_stop_inputs.csv',index=False)
    if train_crop is not None:
        train_crop.to_csv(folder/'actual_training_crops.csv',index=False)
    if val_crop is not None and epochs is None:
        val_crop.to_csv(folder/'actual_early_stop_crops.csv',index=False)
    return _research_stage(training,validation,cfg,device,stage,folder,seed,train_crop,val_crop,epochs)


def optimized_crossfit(fitting,cfg,device,folder,epochs,seed):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    inner=assign_folds(fitting,replace(cfg,N_FOLDS=cfg.INNER_CROP_FOLDS,SEED=seed))
    predictions,audit=[],[]
    for fold in sorted(inner.fold.unique()):
        tr=inner.loc[inner.fold!=fold].reset_index(drop=True)
        va=inner.loc[inner.fold==fold].reset_index(drop=True)
        assert not set(tr.group)&set(va.group)
        model,_=optimized_stage(tr,None,cfg,device,'coarse',folder/f'inner_{fold}',seed+int(fold),epochs=epochs)
        predictions.append(predict_research(model,va,cfg,device,'coarse'))
        audit.append({'inner_fold':int(fold),'fit_ids':tr.image_id.tolist(),'held_ids':va.image_id.tolist(),
                      'fit_groups':tr.group.tolist(),'held_groups':va.group.tolist()})
        _research_cleanup(model);del model
    coarse=pd.concat(predictions,ignore_index=True).set_index('image_id').loc[fitting.image_id].reset_index()
    assert coarse.image_id.is_unique and len(coarse)==len(fitting)
    coarse.to_csv(folder/'training_oof_centres.csv',index=False)
    json_save(audit,folder/'crossfit_audit.json')
    return make_roi_spec(coarse,fitting,cfg,mode='fixed')


def optimized_inference(fine,frame,coarse,cfg,device):
    """No label-dependent crop repair; combine three predefined fields of view."""
    frame=frame[['image_id','path','width','height']].copy()
    center=coarse.set_index('image_id').loc[frame.image_id].reset_index()
    views=[]
    for fraction in cfg.FINE_VALIDATION_ROI_FRACTIONS:
        roi=make_roi_spec(center,frame,cfg,mode='fixed',fixed_fraction=float(fraction))
        pred=predict_research(fine,frame,cfg,device,'fine',roi)
        assert pred.image_id.tolist()==frame.image_id.tolist()
        views.append(pred.assign(roi_fraction=float(fraction)))
    all_views=pd.concat(views,ignore_index=True)
    mean=all_views.groupby('image_id',sort=False)[['x','y']].mean().loc[frame.image_id].reset_index()
    blend=mean.copy()
    weight=float(cfg.COARSE_BLEND_WEIGHT)
    blend[['x','y']]=weight*center[['x','y']].to_numpy()+(1-weight)*mean[['x','y']].to_numpy()
    return mean,blend,all_views


def run_optimized(config):
    cfg=replace(config)
    if cfg.SMOKE_TEST:
        raise ValueError("This release is a genuine training/inference run; use a separate smoke output for debugging.")
    if cfg.N_FOLDS!=5 or cfg.COARSE_BLEND_WEIGHT!=0.5:
        raise ValueError("This frozen experiment prespecifies five folds and an equal coarse/fine blend.")
    if tuple(cfg.ROI_FRACTIONS)!=tuple(cfg.FINE_VALIDATION_ROI_FRACTIONS):
        raise ValueError("Training and inference scale sets must agree.")
    if not torch.cuda.is_available():
        raise RuntimeError("A working GPU is required for this training run.")
    seed_everything(cfg.SEED);cv2.setNumThreads(0)
    device=torch.device('cuda');out=Path(cfg.OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    cfg.PRETRAINED_WEIGHTS_PATH=resolve_research_weights(cfg)
    train,test,sample,layout,metadata=prepare_tables(cfg)
    assert train.visible.all() and not set(train.image_id)&set(test.image_id)
    if set(train.pixel_hash)&set(test.pixel_hash):
        raise ValueError("Exact train/test pixel duplicates require investigation before training.")
    train=assign_folds(train,cfg)
    train.to_csv(out/'folds.csv',index=False);test.to_csv(out/'test_manifest.csv',index=False)
    identity={'config':asdict(cfg),'source':research_runtime_code_digest(),'environment':research_training_environment(),
              'train':train[['image_id','pixel_hash','x','y','group','fold']].to_dict('records'),
              'test':test[['image_id','pixel_hash']].to_dict('records')}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())['signature']!=signature:
        raise ValueError('Output folder belongs to another configuration/code/data; choose a new folder.')
    json_save({'config':asdict(cfg),'signature':signature,'protocol':'Full outer training fold refit; no uncertainty calibration; fixed 50/50 coarse and multiscale fine fusion.',
               'test_labels_used':False,'submission_method_prespecified':'BLEND'},config_path)
    json_save(research_training_environment(),out/'environment.json')
    oof_all={k:[] for k in ['COARSE','FINE_MULTISCALE','BLEND']};test_all={k:[] for k in oof_all}
    for fold in sorted(train.fold.unique()):
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True)
        marker=folder/'complete.json'
        if marker.exists() and cfg.REUSE_CHECKPOINTS:
            record=json.loads(marker.read_text())
            if record['signature']!=signature:raise ValueError('Fold identity mismatch.')
            research_verify_cache(folder,record['files'])
            for method in oof_all:
                oof_all[method].append(pd.read_csv(folder/f'{method}_oof.csv',dtype={'image_id':str,'group':str}))
                test_all[method].append(pd.read_csv(folder/f'{method}_test.csv',dtype={'image_id':str}))
            print('Reuse completed outer fold',fold,flush=True)
            continue
        outer=train.loc[train.fold!=fold].reset_index(drop=True)
        validation=train.loc[train.fold==fold].reset_index(drop=True)
        seed=cfg.SEED+int(fold)*1000
        fitting,stopping=research_group_holdout(outer,cfg.EARLY_STOP_FRACTION,seed+1)
        assert not set(outer.group)&set(validation.group)
        assert not set(fitting.group)&set(stopping.group)
        assert not (set(outer.image_id)|set(validation.image_id))&set(test.image_id)
        roles=pd.concat([fitting[['image_id','group']].assign(role='model_fit'),
                         stopping[['image_id','group']].assign(role='early_stop'),
                         validation[['image_id','group']].assign(role='outer_validation')])
        roles.to_csv(folder/'split_roles.csv',index=False)
        print(f'OUTER FOLD {fold}: fit={len(fitting)}, stop={len(stopping)}, final_train={len(outer)}, outer_validation={len(validation)}',flush=True)
        coarse_select,ci=optimized_stage(fitting,stopping,cfg,device,'coarse',folder/'coarse_select',seed+10)
        stop_prediction=predict_research(coarse_select,stopping,cfg,device,'coarse')
        stop_roi=make_roi_spec(stop_prediction,stopping,cfg,mode='fixed')
        stop_prediction.to_csv(folder/'early_stop_coarse.csv',index=False)
        _research_cleanup(coarse_select);del coarse_select
        fit_roi=optimized_crossfit(fitting,cfg,device,folder/'crop_crossfit',ci['best_epoch'],seed+100)
        full_roi=pd.concat([fit_roi,stop_roi],ignore_index=True)
        assert set(full_roi.image_id)==set(outer.image_id) and full_roi.image_id.is_unique
        fine_select,fi=optimized_stage(fitting,stopping,cfg,device,'fine',folder/'fine_select',seed+20,train_crop=fit_roi,val_crop=stop_roi)
        _research_cleanup(fine_select);del fine_select
        coarse,_=optimized_stage(outer,None,cfg,device,'coarse',folder/'coarse_final',seed+30,epochs=ci['best_epoch'])
        coarse_val=predict_research(coarse,validation,cfg,device,'coarse')
        coarse_test=predict_research(coarse,test,cfg,device,'coarse')
        _research_cleanup(coarse);del coarse
        fine,_=optimized_stage(outer,None,cfg,device,'fine',folder/'fine_final',seed+40,epochs=fi['best_epoch'],train_crop=full_roi)
        metrics=[]
        for kind,frame,cp in [('oof',validation,coarse_val),('test',test,coarse_test)]:
            fp,bp,views=optimized_inference(fine,frame,cp,cfg,device)
            views.to_csv(folder/f'fine_scales_{kind}.csv',index=False)
            for method,pred in [('COARSE',cp),('FINE_MULTISCALE',fp),('BLEND',bp)]:
                if kind=='oof':
                    part=_research_attach_truth(pred,frame,int(fold),method)
                    oof_all[method].append(part)
                    metrics.append({'fold':int(fold),'method':method,**localization_metrics(pred,frame)})
                else:
                    part=pred.assign(fold=int(fold),method=method);test_all[method].append(part)
                part.to_csv(folder/f'{method}_{kind}.csv',index=False)
        _research_cleanup(fine);del fine
        pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
        record={'signature':signature,'fold':int(fold),'final_training_images':len(outer),
                'coarse_epochs':ci['best_epoch'],'fine_epochs':fi['best_epoch'],'metrics':metrics}
        record['files']=research_cache_manifest(folder,[p for p in folder.iterdir() if p.is_file() and p.name!='complete.json'])
        json_save(record,marker)
        print('COMPLETED OUTER FOLD',fold,flush=True)
    metrics=[];combined={}
    for method in oof_all:
        pred=pd.concat(oof_all[method],ignore_index=True)
        assert len(pred)==len(train) and pred.image_id.is_unique
        pred=pred.set_index('image_id').loc[train.image_id].reset_index()
        pred.to_csv(out/f'oof_{method}.csv',index=False);combined[method]=pred
        metrics.append({'method':method,**localization_metrics(pred,train)})
        tests=pd.concat(test_all[method],ignore_index=True)
        assert (tests.groupby('image_id').size()==5).all()
        tests.to_csv(out/f'test_all_folds_{method}.csv',index=False)
        average=tests.groupby('image_id',as_index=False)[['x','y']].mean()
        ensemble=test[['image_id','width','height']].merge(average,on='image_id',validate='one_to_one')
        ensemble['visibility_prob']=1.0
        ensemble.to_csv(out/f'test_ensemble_{method}.csv',index=False)
        path=cfg.SUBMISSION_PATH if method=='BLEND' else str(out/f'submission_{method}.csv')
        write_submission(sample,layout,ensemble,metadata,replace(cfg,SUBMISSION_PATH=path),.5)
    results=pd.DataFrame(metrics);results.to_csv(out/'ablation_summary.csv',index=False)
    paired_group_bootstrap(combined,train,baseline='COARSE',n_bootstrap=1000,seed=cfg.SEED).to_csv(out/'paired_group_bootstrap.csv',index=False)
    diag=combined['BLEND'].copy();diag['visibility_prob']=1.0;diag['visible_true']=True
    plot_oof_errors(diag,train,.5,cfg,out)
    json_save({'status':'COMPLETE','n_train':len(train),'n_test':len(test),'n_folds':5,'stages_per_fold':7,
               'submission_method':'BLEND','coarse_weight':cfg.COARSE_BLEND_WEIGHT,
               'metrics':metrics,'submission_sha256':research_file_digest(cfg.SUBMISSION_PATH)},out/'run_summary.json')
    print(results.to_string(index=False),flush=True)
    return {'config':cfg,'summary':results,'oof':combined,'train':train,'test':test}


from functools import lru_cache


@lru_cache(maxsize=128)
def _optimized_cached_letterbox(path, size, width, height):
    image=read_rgb(path)
    if image.shape[:2] != (height,width):
        raise ValueError("Image dimensions changed.")
    return letterbox(image,size)


def optimized_coarse_view(path, size, width, height):
    image,transform=_optimized_cached_letterbox(str(path),int(size),int(width),int(height))
    return image.copy(),transform.copy()
