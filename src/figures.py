import argparse
import json

import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data import ROOT, split_items
from src.evaluate import dataset
from src.models import MODELS
from src.train import run_epoch

ASSETS = ROOT / "report_assets"
RES = ROOT / "results"
COL = 3.25  # cvpr single-column width in inches
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order, never cycled or reordered
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])  # one hue, light->dark
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#d03b3b"])  # two poles, neutral midpoint
MUTED = "#898781"
CONDS = ["test", "cifake4k", "ddpm", "jpeg75", "jpeg50", "jpeg30", "blur1", "blur2"]
BARS = ["test", "ddpm", "jpeg50", "blur2"]
NAMES = {("resnet18", "rgb"): "ResNet-18 RGB", ("resnet18", "fft"): "ResNet-18 FFT",
         ("convnext", "rgb"): "ConvNeXt-T RGB", ("convnext", "fft"): "ConvNeXt-T FFT",
         ("twostream", "both"): "Two-stream RGB+FFT", ("twostream", "rgb2"): "Two-stream RGB+RGB"}

mpl.rcParams.update({"font.size": 9, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "figure.dpi": 200,
                     "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})


def label(r):
    m = r.get("mask_lf")
    if pd.notna(m) and m:  # notna first: runs predating --mask_lf have NaN here, and NaN is truthy
        return "ResNet-18 FFT (LF-masked)"
    if not r["log"]:
        return "ResNet-18 FFT (raw mag)"
    if r["gray"]:
        return "ResNet-18 FFT (gray)"
    return NAMES[(r["arch"], r["domain"])]


def load():
    m = pd.DataFrame([json.loads(p.read_text()) for p in sorted(RES.glob("*/metrics.json"))])
    m = m[m.run.str.match(r"^[ea]\d")].copy() if len(m) else m  # drop smoke/probe runs
    if not len(m):
        return m, pd.DataFrame()
    m["model"] = m.apply(label, axis=1)
    m["ablation"] = m.run.str.startswith("a")
    ev = pd.read_csv(RES / "all_evals.csv") if (RES / "all_evals.csv").exists() else pd.DataFrame()
    lf = pd.read_csv(RES / "late_fusion.csv") if (RES / "late_fusion.csv").exists() else pd.DataFrame()
    if len(ev):  # accuracies come from all_evals only; metrics.json supplies labels, never numbers
        ev = ev.drop(columns=["arch", "domain"]).merge(m[["run", "model", "ablation"]], on="run")
    if len(lf):
        ev = pd.concat([ev, lf.assign(model="Late fusion (RGB+FFT)", ablation=False)], ignore_index=True)
    return m, ev


def mean_spec(paths):
    acc = 0
    for p in paths:
        acc = acc + spec(p)
    return acc / len(paths)


def spec(p):
    x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)
    return torch.log1p(torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1)).abs()).mean(0).numpy()


def radial(s):
    c = s.shape[0] // 2
    yy, xx = np.mgrid[: s.shape[0], : s.shape[1]]
    r = np.hypot(yy - c, xx - c).astype(int)
    return np.bincount(r.ravel(), s.ravel()) / np.bincount(r.ravel())


def sources():
    test = split_items("test")
    d = {"real": [p for p, y in test if y == 0][:2000], "fake": [p for p, y in test if y == 1][:2000]}
    ddpm = sorted((ROOT / "data" / "ddpm_fakes").glob("*.jpg"))
    if ddpm:
        d["ddpm"] = ddpm
    return d


def spectra(src):
    cache = RES / "spectra.npz"
    if cache.exists():
        return dict(np.load(cache))
    d = {k: mean_spec(v) for k, v in src.items()}
    np.savez(cache, **d)
    return d


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(COL, 1.55))
    ax.set(xlim=(0, 10), ylim=(0, 3.2))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.14", linewidth=0.8)
    for x, y, t, fc in [(0.9, 2.3, "32$\\times$32\nimage", "#f0efec"), (3.5, 2.3, "ResNet-18", CAT[0] + "33"),
                        (3.5, 0.6, "FFT $\\to$ fftshift\n$\\log(1+|F|)$", "#f0efec"),
                        (6.2, 0.6, "ResNet-18", CAT[1] + "33"), (8.6, 1.45, "REAL /\nFAKE", "#f0efec")]:
        ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.35), 1.7, 0.7, fc=fc, ec="#c3c2b7", **box))
        ax.text(x, y, t, ha="center", va="center", fontsize=6)
    for a, b in [((1.75, 2.3), (2.65, 2.3)), ((0.9, 1.95), (2.65, 0.75)), ((4.35, 2.3), (7.75, 1.7)),
                 ((4.35, 0.6), (5.35, 0.6)), ((7.05, 0.6), (7.75, 1.2))]:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=6, color=MUTED, lw=0.7))
    ax.text(3.5, 3.0, "spatial arm", fontsize=6, color=CAT[0], ha="center")
    ax.text(4.9, 0.02, "frequency arm (fixed, non-learned front end)", fontsize=6, color=CAT[1], ha="center")
    fig.savefig(ASSETS / "fig1_pipeline.pdf")
    plt.close(fig)


def fig_mean_spectra(s):
    fig, ax = plt.subplots(1, 3, figsize=(COL, 1.45))
    v = max(abs(s["fake"] - s["real"]).max(), 1e-9)
    panels = [("REAL", s["real"], dict(cmap=SEQ)), ("FAKE", s["fake"], dict(cmap=SEQ)),
              ("FAKE $-$ REAL", s["fake"] - s["real"], dict(cmap=DIV, vmin=-v, vmax=v))]
    for a, (t, im, kw) in zip(ax, panels):
        h = a.imshow(im, **kw)
        a.set_title(t, pad=3, fontsize=7)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        fig.colorbar(h, ax=a, fraction=0.046, pad=0.03).ax.tick_params(labelsize=5, length=2)
    fig.savefig(ASSETS / "mean_spectra.pdf")
    plt.close(fig)


def fig_radial(s):
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for (k, t), c in zip([("real", "CIFAKE REAL"), ("fake", "CIFAKE FAKE"), ("ddpm", "DDPM")], CAT):
        if k in s:
            ax.plot(radial(s[k]), label=t, color=c, lw=1.6)
    ax.set(xlabel="radial spatial frequency (px)", ylabel="mean $\\log(1+|F|)$")
    ax.legend(frameon=False)
    fig.savefig(ASSETS / "radial_profile.pdf")
    plt.close(fig)


def fig_samples(src):
    keys = [k for k in ("real", "fake", "ddpm") if k in src]
    fig, ax = plt.subplots(2 * len(keys), 6, figsize=(COL, 1.15 * len(keys)))
    for i, k in enumerate(keys):
        for j, p in enumerate(src[k][:6]):
            ax[2 * i, j].imshow(Image.open(p))
            ax[2 * i + 1, j].imshow(spec(p), cmap=SEQ)
        ax[2 * i, 0].set_ylabel({"real": "REAL", "fake": "FAKE", "ddpm": "DDPM"}[k], fontsize=6)
        ax[2 * i + 1, 0].set_ylabel("spectrum", fontsize=5)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(ASSETS / "samples_grid.pdf")
    plt.close(fig)


def fig_curves(m):
    d = m[(m.seed == 0) & m.model.isin(["ResNet-18 RGB", "ResNet-18 FFT"])]
    if not len(d):
        return print("skip curves.pdf: no rgb/fft seed-0 runs yet")
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    for (_, r), c in zip(d.iterrows(), CAT):
        cv = pd.read_csv(RES / r.run / "curve.csv")
        ax.plot(cv.epoch, cv.train_acc, color=c, lw=1.5, label=r.model)
        ax.plot(cv.epoch, cv.val_acc, color=c, lw=1.5, ls=(0, (3, 2)))  # dashed = val; the gap is the overfit signal
    ax.set(xlabel="epoch", ylabel="accuracy")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.02, 0.04, "solid = train, dashed = val", transform=ax.transAxes, fontsize=6, color=MUTED)
    fig.savefig(ASSETS / "curves.pdf")
    plt.close(fig)


def fig_bars(ev):
    d = ev[ev.condition.isin(BARS)]  # ablations included: a4 on ddpm is the paper's strongest single result
    if not len(d):
        return print("skip bars_generalization.pdf: no eval rows yet")
    piv = d.pivot_table(index="model", columns="condition", values="auroc", aggfunc="mean").reindex(columns=BARS)
    piv = piv.sort_values("ddpm")  # worst cross-generator at the bottom: a4 anchors the story
    y = np.arange(len(piv))
    w = 0.8 / len(BARS)
    # horizontal: ten model names will not fit as vertical tick labels at 3.25in
    fig, ax = plt.subplots(figsize=(COL, 0.42 * len(piv) + 0.6))
    for i, (c, col) in enumerate(zip(BARS, CAT)):
        ax.barh(y + i * w - 0.4 + w / 2, piv[c], w * 0.88, label=c, color=col)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.text(0.505, len(piv) - 0.35, "chance", fontsize=6, color=MUTED)
    ax.set_yticks(y, piv.index, fontsize=6)
    ax.set(xlabel="AUROC", xlim=(0, 1.02))
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=4, columnspacing=0.8, handlelength=1.0, loc="lower center",
              bbox_to_anchor=(.5, 1.0), fontsize=6)
    fig.savefig(ASSETS / "bars_generalization.pdf")
    plt.close(fig)


def pm(g, k):
    return f"{g[k].mean():.4f}" + (f" $\\pm$ {g[k].std():.4f}" if g[k].count() > 1 else "")


def tables(ev):
    if not len(ev):
        return print("skip tables: all_evals.csv empty")
    t = ev[ev.condition == "test"]
    for name, d in [("table1_main", t[~t.ablation]), ("table3_ablations", ev[ev.ablation & (ev.condition.isin(
            ["test", "ddpm"]))])]:
        if name == "table1_main":
            rows = [{"Model": k, "Seeds": len(g), "Acc": pm(g, "acc"), "F1": pm(g, "f1"), "AUROC": pm(g, "auroc")}
                    for k, g in d.groupby("model")]
        else:
            rows = [{"Model": k, "Condition": c, "Acc": pm(g, "acc"), "F1": pm(g, "f1"), "AUROC": pm(g, "auroc")}
                    for (k, c), g in d.groupby(["model", "condition"])]
        if rows:
            (ASSETS / f"{name}.tex").write_text(pd.DataFrame(rows).to_latex(index=False, escape=False))
    # auroc only: accuracy under corruption is threshold-confounded, so showing both contradicts the text
    t2 = ev.pivot_table(index="model", columns="condition", values="auroc", aggfunc="mean")
    t2 = t2.reindex(columns=[c for c in CONDS if c in ev.condition.unique()])
    (ASSETS / "table2_robustness.tex").write_text(t2.to_latex(float_format="%.3f", escape=False))


def late_fusion():
    out = RES / "late_fusion.csv"
    cks = [RES / "e1_rgb_s0" / "best.pt", RES / "e2_fft_s0" / "best.pt"]
    if not all(c.exists() for c in cks):
        return print("skip late fusion: e1/e2 seed-0 checkpoints not ready")
    have = set(pd.read_csv(out).condition) if out.exists() else set()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    rows = []
    for cond in [c for c in CONDS if c not in have]:
        ps = []
        for ck in cks:  # both arms see identical item order (shuffle=False), so softmaxes align row-for-row
            k = torch.load(ck, map_location=dev)
            a = k["args"]
            mod = MODELS[a["arch"]](in_ch=1 if a["gray"] and a["domain"] != "rgb" else 3).to(dev)
            mod.load_state_dict(k["model"])
            _, _, y, p = run_epoch(mod, DataLoader(dataset(cond, a), batch_size=512, pin_memory=True), dev,
                                   nn.CrossEntropyLoss())
            ps.append(p)
            del mod
            torch.cuda.empty_cache() if dev == "cuda" else None
        p = np.mean(ps, 0)
        rows.append(dict(run="late_fusion", seed=0, condition=cond, acc=accuracy_score(y, p > 0.5),
                         f1=f1_score(y, p > 0.5), auroc=roc_auc_score(y, p), n=len(y)))
        print(f"late_fusion {cond} acc={rows[-1]['acc']:.4f} auroc={rows[-1]['auroc']:.4f}")
    if rows:
        pd.concat([pd.read_csv(out) if out.exists() else pd.DataFrame(), pd.DataFrame(rows)]).to_csv(out, index=False)


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--fusion", action="store_true")  # opt-in: it needs the gpu, so keep it off while the queue runs
    if q.parse_args().fusion:
        late_fusion()
    ASSETS.mkdir(exist_ok=True)
    src = sources()
    s = spectra(src)
    fig_pipeline()
    fig_mean_spectra(s)
    fig_radial(s)
    fig_samples(src)
    m, ev = load()
    if len(m):
        fig_curves(m)
    fig_bars(ev) if len(ev) else None
    tables(ev)
    print(f"runs={len(m)} eval_rows={len(ev)} -> {sorted(p.name for p in ASSETS.glob('*.pdf'))} "
          f"{sorted(p.name for p in ASSETS.glob('*.tex'))}")


if __name__ == "__main__":
    main()
