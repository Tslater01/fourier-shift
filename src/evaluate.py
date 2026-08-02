import argparse
import csv

import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data import CIFAKE, ROOT, _hash
from src.models import MODELS
from src.train import run_epoch

DDPM_DIR = ROOT / "data" / "ddpm_fakes"
N_DDPM = 4000


PAIRED = ("ddpm", "cifake4k")


def dataset(cond, a):
    ds = CIFAKE("test", domain=a["domain"], condition=None if cond in PAIRED else cond, log=a["log"],
                gray=a["gray"], mask_lf=a.get("mask_lf", 0))  # .get: checkpoints predating a4 have no mask_lf
    if cond in PAIRED:  # swap the item list instead of a second Dataset class: transforms and stats stay identical
        real = sorted((p for p, y in ds.items if y == 0), key=_hash)[:N_DDPM]
        if cond == "ddpm":
            fake = sorted(DDPM_DIR.glob("*.jpg"))
            assert len(fake) == N_DDPM, f"expected {N_DDPM} ddpm fakes, found {len(fake)} — generation incomplete"
        else:  # cifake4k: identical reals, cifake fakes. isolates the generator from the smaller real subset
            fake = sorted((p for p, y in ds.items if y == 1), key=_hash)[:N_DDPM]
        ds.items = [(p, 0) for p in real] + [(p, 1) for p in fake]
    return ds


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--ckpt", required=True)
    q.add_argument("--condition", default="test",
                   choices=["test", "cifake4k", "ddpm", "jpeg75", "jpeg50", "jpeg30", "blur1", "blur2"])
    q.add_argument("--bs", type=int, default=512)
    e = q.parse_args()

    torch.backends.cudnn.benchmark = True  # match train.py's code path exactly; every reported number comes from one
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(e.ckpt, map_location=dev)
    a = ck["args"]  # rebuilt from the checkpoint, so arch/domain/log/gray are never re-specified on the cli
    model = MODELS[a["arch"]](in_ch=1 if a["gray"] and a["domain"] != "rgb" else 3).to(dev)
    model.load_state_dict(ck["model"])
    dl = DataLoader(dataset(e.condition, a), batch_size=e.bs, pin_memory=True)
    _, acc, y, p = run_epoch(model, dl, dev, nn.CrossEntropyLoss())

    row = dict(run=a["run"], arch=a["arch"], domain=a["domain"], seed=a["seed"], condition=e.condition,
               acc=acc, f1=f1_score(y, p > 0.5), auroc=roc_auc_score(y, p), n=len(y))  # auroc unflipped: <0.5 is a result
    out = ROOT / "results" / "all_evals.csv"
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, list(row))
        if new:
            w.writeheader()
        w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
