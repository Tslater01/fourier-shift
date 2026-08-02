import argparse
from pathlib import Path

import torch
from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/ddpm-cifar10-32"
JPEG = dict(quality=75, subsampling=2)  # must match cifake exactly, else the eval detects compression, not the generator


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--n", type=int, default=4000)
    q.add_argument("--steps", type=int, default=50)
    q.add_argument("--bs", type=int, default=250)
    q.add_argument("--seed", type=int, default=1234)
    q.add_argument("--out", default="data/ddpm_fakes")
    a = q.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DDIMPipeline(unet=UNet2DModel.from_pretrained(MODEL),
                        scheduler=DDIMScheduler.from_pretrained(MODEL)).to(dev)
    pipe.set_progress_bar_config(disable=True)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    done = len(list(out.glob("*.jpg")))  # resumable: a killed run keeps what it already sampled
    g = torch.Generator(dev).manual_seed(a.seed)
    while done < a.n:
        bs = min(a.bs, a.n - done)
        imgs = pipe(batch_size=bs, num_inference_steps=a.steps, generator=g).images
        for im in imgs:
            im.save(out / f"ddpm_{done:05d}.jpg", "JPEG", **JPEG)
            done += 1
        print(f"{done}/{a.n}")


if __name__ == "__main__":
    main()
