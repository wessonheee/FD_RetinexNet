import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet2D(nn.Module):
    """
    2D Haar Wavelet Transform implemented using pure PyTorch Conv2d.
    Decomposes an input tensor into 4 subbands:
      - LL: Low-frequency approximation (illumination & coarse structure)
      - LH: High-frequency horizontal details
      - HL: High-frequency vertical details
      - HH: High-frequency diagonal details
    Zero external C++/CUDA dependencies, 100% differentiable and invertible.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        # Haar wavelet filters
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        filters = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
        filters = filters.repeat(in_channels, 1, 1).unsqueeze(1)  # [in_channels * 4, 1, 2, 2]

        # Freeze wavelet filters
        self.register_buffer("forward_weight", filters)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            ll: [B, C, H/2, W/2]
            high: [B, 3*C, H/2, W/2] (concatenation of LH, HL, HH)
        """
        b, c, h, w = x.shape
        # Pad if dimensions are odd
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        out = F.conv2d(x, self.forward_weight, stride=2, groups=self.in_channels)
        out = out.view(b, self.in_channels, 4, out.shape[-2], out.shape[-1])

        ll = out[:, :, 0, :, :]
        lh = out[:, :, 1, :, :]
        hl = out[:, :, 2, :, :]
        hh = out[:, :, 3, :, :]
        high = torch.cat([lh, hl, hh], dim=1)  # [B, 3*C, H/2, W/2]
        return ll, high


class InverseHaarWavelet2D(nn.Module):
    """
    Inverse 2D Haar Wavelet Transform (IDWT).
    Reconstructs the original spatial tensor from LL and High subbands.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        filters = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
        filters = filters.repeat(in_channels, 1, 1).unsqueeze(1)  # [in_channels * 4, 1, 2, 2]

        self.register_buffer("inv_weight", filters)

    def forward(self, ll: torch.Tensor, high: torch.Tensor, orig_size=None):
        """
        Args:
            ll: [B, C, H/2, W/2]
            high: [B, 3*C, H/2, W/2]
        """
        b, c, h_half, w_half = ll.shape
        lh, hl, hh = torch.chunk(high, 3, dim=1)

        # Interleave into [B, C*4, H/2, W/2]
        coeffs = torch.stack([ll, lh, hl, hh], dim=2)  # [B, C, 4, H/2, W/2]
        coeffs = coeffs.view(b, c * 4, h_half, w_half)

        out = F.conv_transpose2d(coeffs, self.inv_weight, stride=2, groups=self.in_channels)
        if orig_size is not None:
            out = out[:, :, :orig_size[0], :orig_size[1]]
        return out


class GatedDenoisingBlock(nn.Module):
    """
    Gated Denoising Block (GDB) for High-Frequency branch:
    Dynamically modulates high-frequency features to distinguish between
    amplified dark noise and real structural gradients.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.conv_in = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1)
        self.dw_conv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2)
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x_norm = self.norm(x)
        x_feat = self.dw_conv(self.conv_in(x_norm))
        # Simple Gate: element-wise product of split channels
        f1, f2 = torch.chunk(x_feat, 2, dim=1)
        gated = f1 * torch.sigmoid(f2)
        out = self.conv_out(gated)
        return res + out


class CrossFrequencyAttentionFusion(nn.Module):
    """
    Cross-Frequency Feature Fusion Module (CFFM):
    Enables bidirectional guidance between low-frequency illumination
    and high-frequency detail components before final synthesis.
    """
    def __init__(self, low_dim: int, high_dim: int, out_dim: int):
        super().__init__()
        self.low_proj = nn.Conv2d(low_dim, out_dim, kernel_size=1)
        self.high_proj = nn.Conv2d(high_dim, out_dim, kernel_size=1)

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim * 2, out_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 2, out_dim * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        )

    def forward(self, f_low: torch.Tensor, f_high: torch.Tensor) -> torch.Tensor:
        f_low = self.low_proj(f_low)
        f_high = self.high_proj(f_high)

        concat = torch.cat([f_low, f_high], dim=1)
        weights = self.attn(concat)
        w_low, w_high = torch.chunk(weights, 2, dim=1)

        modulated = torch.cat([f_low * (1.0 + w_low), f_high * (1.0 + w_high)], dim=1)
        return self.fusion(modulated)
