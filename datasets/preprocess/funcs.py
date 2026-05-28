import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def blur(images: torch.Tensor, kernel_size: int, kernel_sigma: float) -> torch.Tensor:
    # Cast to float32 before blurring — gaussian_blur on uint8 does integer
    # arithmetic and truncates intermediate values, producing wrong results.
    return gaussian_blur(
        images.to(torch.float32),
        kernel_size=[kernel_size, kernel_size],
        sigma=kernel_sigma,
    )


def noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    # Expects float32 input (i.e. coming from blur). Cast back to uint8 once
    # at the end so the clamp is correct on the 0-255 scale.
    eps = torch.randn_like(images) * sigma
    return torch.clamp(images + eps, 0, 255).to(torch.uint8)


def resize_crop(images: torch.Tensor, target_size: int) -> torch.Tensor:
    # images: (N, C, H, W) uint8
    _, _, h, w = images.shape

    scale = target_size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    resized = F.interpolate(
        images.to(torch.float32),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )

    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2

    return (
        resized[:, :, top:top + target_size, left:left + target_size]
        .clamp_(0.0, 255.0)
        .round_()
        .to(torch.uint8)
    )


def tile(images: torch.Tensor, tile_size: int) -> torch.Tensor:
    n, c, h, w = images.shape

    if h % tile_size != 0 or w % tile_size != 0:
        raise ValueError(
            f"Image size ({h}x{w}) must be divisible by tile_size ({tile_size})"
        )

    tiles_h = h // tile_size
    tiles_w = w // tile_size

    # (N, C, tiles_h, tile_size, tiles_w, tile_size)
    tiled = images.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)

    # (N, tiles_h, tiles_w, C, tile_size, tile_size)
    tiled = tiled.permute(0, 2, 3, 1, 4, 5).contiguous()

    # (N, T, C, tile_size, tile_size)
    return tiled.view(n, tiles_h * tiles_w, c, tile_size, tile_size)


def untile(tiles: torch.Tensor, tiles_h: int, tiles_w: int) -> torch.Tensor:
    n, t, c, tile_size, _ = tiles.shape

    assert t == tiles_h * tiles_w, (
        f"Expected {tiles_h * tiles_w} tiles (tiles_h={tiles_h}, tiles_w={tiles_w}), got {t}"
    )

    # (N, tiles_h, tiles_w, C, tile_size, tile_size)
    x = tiles.view(n, tiles_h, tiles_w, c, tile_size, tile_size)

    # (N, C, tiles_h, tile_size, tiles_w, tile_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

    # (N, C, H, W)
    return x.view(n, c, tiles_h * tile_size, tiles_w * tile_size)