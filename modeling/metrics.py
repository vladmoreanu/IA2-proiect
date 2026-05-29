from lighter.metrics import Metric

import torch


def mse(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return torch.mean((targets - outputs) ** 2)


def psnr(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return 20 * torch.log10(1.0 / torch.sqrt(mse(targets, outputs)))


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
        return (self.total_psnr / self.count).item() if self.count > 1 else self.total_psnr.item()