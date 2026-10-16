import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1 proxy)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for PyTorch tensors."""
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _gaussian(self, window_size: int, sigma: float):
        gauss = torch.exp(
            -torch.tensor([(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)])
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return nn.Parameter(window, requires_grad=False)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1 = img1.float()
        img2 = img2.float()
        window = self.window.to(dtype=img1.dtype, device=img1.device)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1.0 - ssim_map.mean()


class FrequencyAmplitudeLoss(nn.Module):
    """
    Normalized Fourier Frequency Amplitude Loss:
    Aligns global illumination and spectral energy distribution in frequency domain.
    Uses norm='ortho' to guarantee Parseval energy balance with spatial loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fft_pred = torch.fft.rfft2(pred, norm="ortho")
        fft_target = torch.fft.rfft2(target, norm="ortho")

        # Stable magnitude with epsilon protection against singularity at zero
        amp_pred = torch.sqrt(fft_pred.real.pow(2) + fft_pred.imag.pow(2) + 1e-8)
        amp_target = torch.sqrt(fft_target.real.pow(2) + fft_target.imag.pow(2) + 1e-8)

        return F.l1_loss(amp_pred, amp_target)


class ColorCosineLoss(nn.Module):
    """
    Cosine Color Loss:
    Measures the angular deviation of color vectors in RGB space.
    Constrains hue and saturation fidelity, preventing low-light color casts.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        norm_pred = torch.norm(pred, dim=1, keepdim=True) + self.eps
        norm_target = torch.norm(target, dim=1, keepdim=True) + self.eps
        cosine = torch.sum(pred * target, dim=1, keepdim=True) / (norm_pred * norm_target)
        return torch.mean(1.0 - cosine)


class CompoundEnhancementLoss(nn.Module):
    """
    Balanced compound loss:
      - Charbonnier Reconstruction Loss (pixel-level structural fidelity)
      - SSIM Loss (structural contrast & local luminance matching)
      - Frequency Amplitude Loss (orthonormal spectral energy alignment)
      - Cosine Color Loss (RGB chromatic angular fidelity & white balance)
    """
    def __init__(
        self,
        lambda_ssim: float = 0.5,
        lambda_freq: float = 0.1,
        lambda_color: float = 0.2,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_loss = SSIMLoss()
        self.freq_loss = FrequencyAmplitudeLoss()
        self.color_loss = ColorCosineLoss()

        self.lambda_ssim = lambda_ssim
        self.lambda_freq = lambda_freq
        self.lambda_color = lambda_color

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()

        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        l_color = self.color_loss(pred, target)

        total = l_char + self.lambda_ssim * l_ssim + self.lambda_freq * l_freq + self.lambda_color * l_color
        loss_dict = {
            "total": total.item(),
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "freq": l_freq.item(),
            "color": l_color.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    p = torch.rand(2, 3, 256, 256)
    t = torch.rand(2, 3, 256, 256)
    crit = CompoundEnhancementLoss()
    total, d = crit(p, t)
    print("Loss calculation smoke test:", d)
