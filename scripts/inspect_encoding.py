import io
from collections import Counter

from PIL import Image, JpegImagePlugin

from src.data import CIFAKE_DIR

N = 400


def qtable_ref():
    ref = {}
    img = Image.new("RGB", (32, 32))
    for q in range(1, 101):
        for sub in (0, 1, 2):
            b = io.BytesIO()
            img.save(b, "JPEG", quality=q, subsampling=sub)
            b.seek(0)
            ref[tuple(tuple(t) for t in Image.open(b).quantization.values())] = (q, sub)
    return ref


def probe(split, cls, ref):
    paths = sorted((CIFAKE_DIR / split / cls).iterdir())
    paths = paths[:: max(1, len(paths) // N)][:N]
    fmt, mode, size, qual, sub, nbytes = Counter(), Counter(), Counter(), Counter(), Counter(), 0
    for p in paths:
        im = Image.open(p)
        fmt[im.format] += 1
        mode[im.mode] += 1
        size[im.size] += 1
        nbytes += p.stat().st_size
        key = tuple(tuple(t) for t in im.quantization.values())
        q, s = ref.get(key, (None, None))
        qual[q] += 1
        sub[JpegImagePlugin.get_sampling(im)] += 1
    print(f"{split}/{cls:4s} n={len(paths)}  fmt={dict(fmt)} mode={dict(mode)} size={dict(size)} "
          f"subsampling={dict(sub)}  mean_bytes={nbytes/len(paths):.0f}")
    print(f"{'':16s}quality={dict(qual.most_common(5))}")


def main():
    ref = qtable_ref()
    for split in ("train", "test"):
        for cls in ("REAL", "FAKE"):
            probe(split, cls, ref)


if __name__ == "__main__":
    main()
