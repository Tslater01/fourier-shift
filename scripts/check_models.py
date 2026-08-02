import time

import torch
from torch import nn

from src.models import MODELS

SPECS = [("resnet18", 512, 1, 3), ("resnet18", 512, 1, 1), ("convnext", 64, 1, 3), ("twostream", 256, 2, 3)]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}\n")
    for arch, bs, arms, in_ch in SPECS:
        m = MODELS[arch](in_ch=in_ch).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        xs = [torch.randn(bs, 3 if (arms == 2 and i == 0) else in_ch, 32, 32, device=dev) for i in range(arms)]
        y = torch.randint(0, 2, (bs,), device=dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(3):  # fwd+bwd: forward-only vram would badly undercount what the queue needs
            with torch.autocast(dev, torch.float16, enabled=dev == "cuda"):
                out = m(*xs)
                loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        vram = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        print(f"{arch:10s} in_ch={in_ch} bs={bs:4d} arms={arms}  params={sum(p.numel() for p in m.parameters())/1e6:6.2f}M "
              f"out={tuple(out.shape)} loss={loss.item():.3f}  peak_vram={vram:.2f}GB  {(time.perf_counter()-t)/3*1000:6.0f} ms/step")
        del m, opt, xs, out, loss
        torch.cuda.empty_cache() if dev == "cuda" else None


if __name__ == "__main__":
    main()
