from collections import Counter

from torch.utils.data import DataLoader

from src.data import CIFAKE


def show(dom, **kw):
    ds = CIFAKE("train", domain=dom, limit=4000, **kw)
    *arms, y = next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)))
    print(f"\n[{dom}{''.join(f' {k}={v}' for k, v in kw.items())}]  n={len(ds)}  arms={len(arms)}"
          f"  split labels={dict(sorted(Counter(l for _, l in ds.items).items()))}")
    for i, a in enumerate(arms):
        print(f"  arm{i}: shape={tuple(a.shape)} dtype={a.dtype} min={a.min():+8.3f} max={a.max():+8.3f} "
              f"mean={a.mean():+.4f} std={a.std():.4f}")
    print(f"  y: dtype={y.dtype} counts={dict(sorted(Counter(y.tolist()).items()))} vals={y.tolist()}")


def main():
    for dom in ("rgb", "fft", "both"):
        show(dom)
    show("fft", log=False)
    show("fft", gray=True)

    a, b = CIFAKE("val", domain="rgb"), CIFAKE("val", domain="fft")
    pa, pb = [p for p, _ in a.items], [p for p, _ in b.items]
    assert pa == pb, "val split differs across instantiations"
    tr = {p for p, _ in CIFAKE("train", domain="rgb").items}
    assert not tr & set(pa), "train/val overlap"
    print(f"\nval split: n={len(pa)} identical across 2 instantiations, disjoint from train (n={len(tr)}), "
          f"labels={dict(sorted(Counter(l for _, l in a.items).items()))}")
    print(f"first 3 val files: {[p.parent.name + '/' + p.name for p in pa[:3]]}")
    print(f"test: n={len(CIFAKE('test', domain='rgb'))}  jpeg50 corrupt={CIFAKE('test', 'rgb', condition='jpeg50').corrupt}")


if __name__ == "__main__":
    main()
