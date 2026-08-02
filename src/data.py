import hashlib
import io
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
CIFAKE_DIR = ROOT / "data" / "cifake"
STATS_FILE = ROOT / "data" / "fft_stats.json"
CLASSES = {"REAL": 0, "FAKE": 1}  # fixed here; ImageFolder's alphabetical order would invert it
VAL_N = 5000
SPLIT_KEY = b"1234"  # constant split salt, deliberately independent of --seed
STATS_N = 10000


class FFTTransform:
    def __init__(self, mean=None, std=None, log=True, mask_lf=0):
        self.mean, self.std, self.log, self.mask_lf, self.disc = mean, std, log, mask_lf, None

    def __call__(self, x):
        f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))  # dc to center: upsampling artifacts become centered rings
        mag = torch.log1p(f.abs()) if self.log else f.abs()
        if self.mask_lf:  # a4: kill the central disc *before* standardizing, so dc/brightness cannot carry the class
            h, w = mag.shape[-2:]
            if self.disc is None:
                yy, xx = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(w) - w // 2, indexing="ij")
                self.disc = (yy.square() + xx.square() > self.mask_lf ** 2).float()
            mag = mag * self.disc
        if self.mean is not None:
            mag = (mag - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        return mag


class Jpeg:
    def __init__(self, q):
        self.q = q

    def __call__(self, img):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=self.q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def corruption(name):
    if name in (None, "test", "ddpm"):
        return None
    if name.startswith("jpeg"):
        return Jpeg(int(name[4:]))
    sigma = float(name[4:])
    return T.GaussianBlur(2 * math.ceil(2 * sigma) + 1, sigma)


def _index(split):
    return [(p, y) for c, y in CLASSES.items() for p in sorted((CIFAKE_DIR / split / c).iterdir())]


def _hash(p):
    return hashlib.blake2b(f"{p.parent.name}/{p.name}".encode(), key=SPLIT_KEY, digest_size=8).digest()


def _val_paths(items):
    per = VAL_N // len(CLASSES)
    return {p for y in CLASSES.values()
            for p in sorted((q for q, yy in items if yy == y), key=_hash)[:per]}


def split_items(split):
    items = _index("test" if split == "test" else "train")
    if split != "test":
        val = _val_paths(items)
        items = [it for it in items if (it[0] in val) == (split == "val")]
    return items


def _subsample(items, limit):
    out = []
    for y in CLASSES.values():  # strided within each class, so --limit stays exactly balanced
        cls = [it for it in items if it[1] == y]
        per = limit // len(CLASSES)
        out += cls[:: max(1, len(cls) // per)][:per]
    return out


def _pipe(kind, mean=None, std=None):
    if kind == "rgb":
        return T.Compose([T.ToTensor()] + ([T.Normalize(mean, std)] if mean is not None else []))
    parts = kind.split("_")
    scale, ch = parts[1], parts[2]
    return T.Compose(([T.Grayscale(1)] if ch == "gray" else [])
                     + [T.ToTensor(), FFTTransform(mean, std, scale == "log", int(parts[3][1:]) if len(parts) > 3 else 0)])


def stats(kind):
    cache = json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else {}
    if kind not in cache:
        items = split_items("train")
        items = items[:: max(1, len(items) // STATS_N)]  # strided: both classes, no rng
        pipe = _pipe(kind)
        s1 = s2 = n = 0
        for p, _ in items:  # running moments; never stacks the sample
            x = pipe(Image.open(p).convert("RGB"))
            s1 = s1 + x.sum((1, 2))
            s2 = s2 + (x * x).sum((1, 2))
            n += x.shape[1] * x.shape[2]
        mean = s1 / n
        cache[kind] = [mean.tolist(), (s2 / n - mean ** 2).clamp_min(0).sqrt().tolist()]
        STATS_FILE.write_text(json.dumps(cache, indent=2))
    m, s = cache[kind]
    return torch.tensor(m), torch.tensor(s)


class CIFAKE(Dataset):
    def __init__(self, split, domain="rgb", limit=None, condition=None, log=True, gray=False, mask_lf=0):
        items = split_items(split)
        if limit:
            items = _subsample(items, limit)
        # mask suffix only when set, so the mask_lf=0 stats cache keys stay byte-identical to every earlier run
        fft_kind = f"fft_{'log' if log else 'raw'}_{'gray' if gray else 'rgb'}" + (f"_m{mask_lf}" if mask_lf else "")
        # rgb2 duplicates the rgb arm: the a3 capacity control, so a two-stream gain is not just 2x the parameters
        kinds = {"rgb": ["rgb"], "fft": [fft_kind], "both": ["rgb", fft_kind], "rgb2": ["rgb", "rgb"]}[domain]
        self.items = items
        self.arms = [_pipe(k, *stats(k)) for k in kinds]  # built here, not in __getitem__: windows spawn pickles these
        self.flip = split == "train"
        self.corrupt = corruption(condition)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.corrupt is not None:
            img = self.corrupt(img)
        if self.flip and torch.rand(()) < 0.5:
            img = TF.hflip(img)  # flip the image, so both arms of `both` see the same view
        return (*(a(img) for a in self.arms), y)
