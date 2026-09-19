"""Image distortion V' = D(V) (paper Eq. 2).

The paper writes V' = V + N(0, sigma^2) + Poisson(lambda) with sigma=0.07, lambda=70.
Adding Poisson(70) to a [0,1] image literally would saturate it, so we follow the
VASE code (Liao et al., MICCAI 2025; paper ref [19]), which uses exactly these values:

    x = clamp(x + N(0, 0.07^2), 0, 1)          # AddGaussianNoise(std=0.07)
    x = clamp(Poisson(clamp(70 x, 0, 255)) / 70, 0, 1)   # AddPoissonNoise(scale=70)
    PIL <- (255 x).byte()                       # torchvision ToPILImage truncates

Noise is applied per RGB channel on the resized image, as in VASE.
mode="vase_full" additionally applies VASE's geometric/colour jitter (ablation only;
the VGS paper mentions only Gaussian + Poisson noise).
"""
from __future__ import annotations

import zlib

import numpy as np
import torch
from PIL import Image


def stable_seed(key: str, base_seed: int = 0) -> int:
    return (zlib.crc32(key.encode("utf-8")) + base_seed) % (2**31 - 1)


def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def resize(img: Image.Image, size: int | None) -> Image.Image:
    if not size:
        return img
    # torchvision.transforms.Resize((s, s)) default = bilinear
    return img.resize((size, size), Image.BILINEAR)


def distort(img: Image.Image, sigma: float = 0.07, lam: float = 70.0, seed: int = 0,
            mode: str = "gp") -> Image.Image:
    g = torch.Generator().manual_seed(int(seed))
    img = to_rgb(img)
    if mode == "vase_full":
        img = _vase_geometric(img, seed)
    elif mode != "gp":
        raise ValueError(f"unknown noise mode {mode}")

    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # H W C in [0,1]
    if sigma > 0:
        x = (x + torch.randn(x.shape, generator=g) * sigma).clamp(0.0, 1.0)
    if lam > 0:
        x = (torch.poisson((x * lam).clamp(0, 255), generator=g) / lam).clamp(0.0, 1.0)
    arr = (x * 255.0).to(torch.uint8).numpy()   # truncation, like ToPILImage
    return Image.fromarray(arr, mode="RGB")


def _vase_geometric(img: Image.Image, seed: int) -> Image.Image:
    import torchvision.transforms as T  # optional dependency
    torch.manual_seed(seed)
    w, h = img.size
    return T.Compose([
        T.RandomResizedCrop(size=(h, w), scale=(0.9, 1.0)),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])(img)


def prepare_pair(img: Image.Image, key: str, size: int | None = 512, sigma: float = 0.07,
                 lam: float = 70.0, mode: str = "gp", base_seed: int = 0):
    """Returns (original, distorted), both resized the same way. Distortion is seeded per sample."""
    orig = resize(to_rgb(img), size)
    dist = distort(orig, sigma=sigma, lam=lam, seed=stable_seed(key, base_seed), mode=mode)
    return orig, dist
