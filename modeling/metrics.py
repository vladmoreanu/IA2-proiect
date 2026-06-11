from lighter.metrics import Metric

import torch
import torch.nn.functional as F


def mse(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return torch.mean((targets - outputs) ** 2)


def l1(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(outputs - targets))


def combined(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return 0.8 * l1(targets, outputs) + 0.2 * mse(targets, outputs)


def psnr(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(1.0 / torch.sqrt(mse(targets, outputs)))


def fast_ssim(
    targets: torch.Tensor,
    outputs: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
):

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    pool = F.avg_pool2d

    mu_x = pool(outputs, window_size, 1, window_size // 2)
    mu_y = pool(targets, window_size, 1, window_size // 2)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = pool(outputs * outputs, window_size, 1, window_size // 2) - mu_x2
    sigma_y2 = pool(targets * targets, window_size, 1, window_size // 2) - mu_y2
    sigma_xy = pool(outputs * targets, window_size, 1, window_size // 2) - mu_xy

    ssim_map = (
        (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )

    return ssim_map.mean()


def combined_L1_SSIM(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return 0.8 * l1(targets, outputs) + 0.2 * (1 - fast_ssim(targets, outputs))


class PSNR(Metric):
    def __init__(self):
        self.count = None
        self.total_psnr = None
        self.name = "psnr"

    def reset(self):
        self.total_psnr = 0
        self.count = 0

    def update(self, targets, outputs):
        self.total_psnr += psnr(targets, outputs).detach()
        self.count += 1

    def result(self):
        return (self.total_psnr / self.count).item() if self.count >= 1 else 0


class MSE(Metric):
    def __init__(self):
        self.count = None
        self.total_mse = None
        self.name = "mse"

    def reset(self):
        self.total_mse = 0
        self.count = 0

    def update(self, targets, outputs):
        self.total_mse += mse(targets, outputs).detach()
        self.count += 1

    def result(self):
        return (self.total_mse / self.count).item() if self.count >= 1 else 0


class Combined_L1_MSE(Metric):
    def __init__(self):
        self.count = None
        self.total_combined = None
        self.name = "0.8l1+0.2mse"

    def reset(self):
        self.total_combined = 0
        self.count = 0

    def update(self, targets, outputs):
        self.total_combined += combined(targets, outputs).detach()
        self.count += 1

    def result(self):
        return (self.total_combined / self.count).item() if self.count >= 1 else 0

    def __call__(self, outputs, targets):
        return combined(targets, outputs)


class Combined_L1_SSIM(Metric):
    def __init__(self):
        self.count = None
        self.total_combined = None
        self.name = "0.8l1+0.2SSIM"

    def reset(self):
        self.total_combined = 0
        self.count = 0

    def update(self, targets, outputs):
        self.total_combined += combined_L1_SSIM(targets, outputs).detach()
        self.count += 1

    def result(self):
        return (self.total_combined / self.count).item() if self.count >= 1 else 0

    def __call__(self, outputs, targets):
        return combined_L1_SSIM(targets, outputs)
