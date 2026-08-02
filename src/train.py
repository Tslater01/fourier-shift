import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data import CIFAKE
from src.models import MODELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"resnet18": dict(lr=3e-4, bs=512, epochs=8),  # val plateaus by ~ep2; 8 is already generous
            "convnext": dict(lr=5e-5, bs=64, epochs=3),
            "twostream": dict(lr=3e-4, bs=256, epochs=8)}


def loader(split, a, shuffle):
    ds = CIFAKE(split, domain=a.domain, limit=a.limit if split == "train" else None, log=a.log, gray=a.gray,
                mask_lf=a.mask_lf)
    nw = a.workers if split == "train" and len(ds) > 10000 else 0  # windows spawn costs ~20-40s per pool; only a
    return DataLoader(ds, batch_size=a.bs, shuffle=shuffle, num_workers=nw,  # full-size train epoch amortizes it
                      persistent_workers=nw > 0, pin_memory=True)


def run_epoch(model, dl, dev, crit, opt=None, scaler=None):
    model.train(opt is not None)
    tot = n = 0
    ys, ps = [], []
    with torch.set_grad_enabled(opt is not None):
        for batch in dl:
            *xs, y = [t.to(dev, non_blocking=True) for t in batch]
            with torch.autocast(dev, torch.float16, enabled=dev == "cuda"):
                out = model(*xs)  # variable arity: rgb/fft pass one tensor, twostream two. no per-domain branching
                loss = crit(out, y)
            if opt is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += loss.item() * y.numel()
            n += y.numel()
            ys.append(y.cpu())
            ps.append(out.detach().float().softmax(1)[:, 1].cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return tot / n, accuracy_score(y, p > 0.5), y, p


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--arch", default="resnet18", choices=list(MODELS))
    q.add_argument("--domain", default="rgb", choices=["rgb", "fft", "both", "rgb2"])
    q.add_argument("--epochs", type=int)
    q.add_argument("--bs", type=int)
    q.add_argument("--lr", type=float)
    q.add_argument("--wd", type=float, default=0.05)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--limit", type=int)
    q.add_argument("--run", required=True)
    q.add_argument("--log", type=int, default=1)   # 0 = ablation a1, raw magnitude
    q.add_argument("--gray", type=int, default=0)  # 1 = ablation a2
    q.add_argument("--mask_lf", type=int, default=0)  # a4: radius in bins of the masked central low-frequency disc
    q.add_argument("--workers", type=int, default=4)
    a = q.parse_args()
    for k, v in DEFAULTS[a.arch].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.log, a.gray = bool(a.log), bool(a.gray)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / "results" / a.run
    out.mkdir(parents=True, exist_ok=True)
    dtr, dva, dte = loader("train", a, True), loader("val", a, False), loader("test", a, False)
    model = MODELS[a.arch](in_ch=1 if a.gray and a.domain != "rgb" else 3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")
    crit = nn.CrossEntropyLoss()
    print(f"{a.run}: {a.arch}/{a.domain} n_train={len(dtr.dataset)} bs={a.bs} lr={a.lr} epochs={a.epochs} dev={dev}")

    best, t0, curve = -1.0, time.perf_counter(), out / "curve.csv"
    for ep in range(1, a.epochs + 1):
        te = time.perf_counter()
        trl, tra, *_ = run_epoch(model, dtr, dev, crit, opt, scaler)
        vll, vla, *_ = run_epoch(model, dva, dev, crit)
        sched.step()
        dt = time.perf_counter() - te
        new = not curve.exists()
        with curve.open("a", newline="") as f:  # appended each epoch, so killing the process keeps what ran
            w = csv.writer(f)
            if new:
                w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "sec"])
            w.writerow([ep, f"{trl:.4f}", f"{tra:.4f}", f"{vll:.4f}", f"{vla:.4f}", f"{dt:.1f}"])
        if vla > best:
            best = vla
            torch.save({"model": model.state_dict(), "args": vars(a)}, out / "best.pt")
        vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        print(f"ep {ep}/{a.epochs}  train {trl:.4f}/{tra:.4f}  val {vll:.4f}/{vla:.4f}  "
              f"{dt:.1f}s  eta {(a.epochs - ep) * dt / 60:.1f}m  vram {vram:.2f}GB")

    model.load_state_dict(torch.load(out / "best.pt")["model"])
    _, _, y, p = run_epoch(model, dte, dev, crit)
    mins = (time.perf_counter() - t0) / 60
    vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0  # after the test pass, not the loop-local
    m = dict(run=a.run, arch=a.arch, domain=a.domain, seed=a.seed, val_acc=best,
             acc=accuracy_score(y, p > 0.5), f1=f1_score(y, p > 0.5), auroc=roc_auc_score(y, p),
             n=len(y), minutes=mins, peak_vram_gb=vram,
             **{k: getattr(a, k) for k in ("epochs", "bs", "lr", "wd", "limit", "log", "gray", "mask_lf")})
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"done in {mins:.1f}m  test acc {m['acc']:.4f}  f1 {m['f1']:.4f}  auroc {m['auroc']:.4f}")


if __name__ == "__main__":
    main()
