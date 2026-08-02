import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data import ROOT, split_items

ASSETS = ROOT / "report_assets"


def mean_spec(paths):
    acc = 0
    for p in paths:
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)
        acc = acc + torch.log1p(torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)).abs())
    return (acc / len(paths)).mean(0).numpy()


def radial(s):
    c = s.shape[0] // 2
    yy, xx = np.mgrid[: s.shape[0], : s.shape[1]]
    r = np.hypot(yy - c, xx - c).astype(int)
    return np.bincount(r.ravel(), s.ravel()) / np.bincount(r.ravel())


def main():
    ASSETS.mkdir(exist_ok=True)
    ddpm = sorted((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/ddpm_preview")).glob("*.jpg"))
    test = split_items("test")
    n = len(ddpm)
    real = [p for p, y in test if y == 0][:2000]
    fake = [p for p, y in test if y == 1][:2000]

    fig, ax = plt.subplots(4, 8, figsize=(10, 5.2))
    for a, p in zip(ax.ravel(), ddpm[:32]):
        a.imshow(Image.open(p))
        a.axis("off")
    fig.suptitle(f"DDPM samples (n={n}, DDIM 50 steps, jpeg q75 4:2:0)")
    fig.tight_layout()
    fig.savefig(ASSETS / "ddpm_contact_sheet.png", dpi=160)

    specs = {f"CIFAKE REAL (n={len(real)})": mean_spec(real),
             f"CIFAKE FAKE (n={len(fake)})": mean_spec(fake),
             f"DDPM (n={n})": mean_spec(ddpm)}
    vmin = min(s.min() for s in specs.values())
    vmax = max(s.max() for s in specs.values())
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for a, (k, s) in zip(ax, specs.items()):
        im = a.imshow(s, cmap="viridis", vmin=vmin, vmax=vmax)  # shared scale: panels are only comparable if identical
        a.set_title(k, fontsize=10)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    for k, s in specs.items():
        ax[3].plot(radial(s), label=k, lw=1.4)
    ax[3].set(xlabel="radial frequency (px)", ylabel="mean log(1+|F|)", title="azimuthal average")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "mean_spectra_preview.png", dpi=160)

    for k, s in specs.items():
        print(f"{k:24s} mean={s.mean():.4f} dc={s[16, 16]:.3f} corner={s[0, 0]:.3f} nyquist_ratio={radial(s)[-1]/s.mean():.3f}")


if __name__ == "__main__":
    main()
