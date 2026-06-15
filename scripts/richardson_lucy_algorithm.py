import torch
import torch.nn.functional as F
import numpy as np
import cv2

def richardson_lucy(image, kernel, iters=30, device='cuda'):
    image = image.to(device)
    kernel = kernel.to(device)

    channels = image.shape[1]

    if kernel.shape[0] == 1 and channels > 1:
        kernel = kernel.repeat(channels, 1, 1, 1)

    kernel_mirror = torch.flip(kernel, dims=[2, 3])

    kh, kw = kernel.shape[2], kernel.shape[3]
    pad_h, pad_w = kh // 2, kw // 2
    padding = (pad_w, pad_w, pad_h, pad_h)

    x = image.clone()

    for _ in range(iters):
        x_padded = F.pad(x, padding, mode='reflect')
        reblurred = F.conv2d(x_padded, kernel, groups=channels)

        relative_blur = image / (reblurred + 1e-12)
        rel_blur_padded = F.pad(relative_blur, padding, mode='reflect')

        error_correction = F.conv2d(rel_blur_padded, kernel_mirror, groups=channels)
        x = x * error_correction

        x = torch.clamp(x, 0.0, 1.0)

    return x


def get_gaussian_kernel(kernel_size=15, sigma=3.0):
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))

    kernel_2d = torch.outer(g, g)

    kernel_2d = kernel_2d / kernel_2d.sum()

    return kernel_2d.view(1, 1, kernel_size, kernel_size)


def main_pipeline(image_tensor, kernel_tensor, num_iters=30):
    device = image_tensor.device

    img_cpu = image_tensor.detach().cpu().squeeze(0)

    img_np = img_cpu.permute(1, 2, 0).numpy()

    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    img_denoised = cv2.bilateralFilter(img_uint8, 9, 75, 75)

    img_denoised_float = img_denoised.astype(np.float32) / 255.0

    tensor_curatat = torch.from_numpy(img_denoised_float).permute(2, 0, 1)

    tensor_curatat = tensor_curatat.unsqueeze(0).to(device)

    tensor_final = richardson_lucy(tensor_curatat, kernel_tensor, iters=num_iters)

    return tensor_final



