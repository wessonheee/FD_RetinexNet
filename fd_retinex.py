import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules.frequency_ops import (
        HaarWavelet2D,
        InverseHaarWavelet2D,
        GatedDenoisingBlock,
        CrossFrequencyAttentionFusion,
    )
except (ImportError, ValueError):
    try:
        from modules.frequency_ops import (
            HaarWavelet2D,
            InverseHaarWavelet2D,
            GatedDenoisingBlock,
            CrossFrequencyAttentionFusion,
        )
    except (ImportError, ValueError):
        from models.modules.frequency_ops import (
            HaarWavelet2D,
            InverseHaarWavelet2D,
            GatedDenoisingBlock,
            CrossFrequencyAttentionFusion,
        )


class ResBlock(nn.Module):
    """Residual block with GroupNorm."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class MultiScaleContextBlock(nn.Module):
    """
    Multi-Scale Global Context Illumination Module (MSCB):
    Combines parallel multi-scale dilated convolutions (1, 2, 4) with
    Global Context Channel Attention to capture ambient scene illumination.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=4, dilation=4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.proj = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = F.leaky_relu(self.conv1(x), 0.2)
        c2 = F.leaky_relu(self.conv2(x), 0.2)
        c3 = F.leaky_relu(self.conv3(x), 0.2)
        f = self.fusion(torch.cat([c1, c2, c3], dim=1))
        scale = self.ca(self.global_pool(f))
        out = self.proj(f * scale)
        return x + out


class LowFrequencyIlluminationBranch(nn.Module):
    """
    L-Branch: Models Retinex illumination and dynamic range.
    Operates on low-frequency approximation subband (LL).
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 64, num_blocks: int = 3, use_mscb: bool = True):
        super().__init__()
        self.head = nn.Conv2d(in_channels, feat_dim, kernel_size=3, padding=1)
        self.mscb = MultiScaleContextBlock(feat_dim) if use_mscb else ResBlock(feat_dim)
        self.body = nn.Sequential(*[ResBlock(feat_dim) for _ in range(num_blocks)])
        self.illum_gain = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim // 2, in_channels, kernel_size=1),
        )
        self.tail = nn.Conv2d(feat_dim, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, ll: torch.Tensor):
        feat = self.head(ll)
        feat = self.mscb(feat)
        feat = self.body(feat)
        gain = 1.0 + F.softplus(self.illum_gain(feat)) * 8.0
        ll_out = ll * gain + self.tail(feat)
        return ll_out, feat, gain


class HighFrequencyDetailBranch(nn.Module):
    """
    H-Branch: Denoises amplified sensor noise and sharpens edge gradients.
    Operates on high-frequency subbands via adaptive soft-gating.
    """
    def __init__(self, in_channels: int = 9, feat_dim: int = 64, num_blocks: int = 3, use_gdb: bool = True):
        super().__init__()
        self.head = nn.Conv2d(in_channels, feat_dim, kernel_size=3, padding=1)
        self.gated_blocks = nn.Sequential(*[
            GatedDenoisingBlock(feat_dim) if use_gdb else ResBlock(feat_dim)
            for _ in range(num_blocks)
        ])
        self.edge_gate = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(feat_dim, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, high: torch.Tensor, gain: torch.Tensor):
        feat = self.gated_blocks(self.head(high))
        gate = self.edge_gate(feat) * 1.5
        repeat_factor = max(1, high.shape[1] // gain.shape[1])
        gain_high = gain.repeat(1, repeat_factor, 1, 1)[:, :high.shape[1], :, :]
        high_clean = high * gain_high * gate + self.tail(feat) * 0.1
        return high_clean, feat


class FDRetinexNet(nn.Module):
    """
    Frequency-Decoupled Retinex Network (FD-RetinexNet)
    Supports ablation flags: use_wavelet, use_gdb, use_mscb, use_cffm.
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 64,
        num_l_blocks: int = 3,
        num_h_blocks: int = 3,
        use_wavelet: bool = True,
        use_gdb: bool = True,
        use_mscb: bool = True,
        use_cffm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_wavelet = use_wavelet
        self.use_cffm = use_cffm

        if self.use_wavelet:
            self.dwt = HaarWavelet2D(in_channels=in_channels)
            self.idwt = InverseHaarWavelet2D(in_channels=in_channels)
            h_channels = in_channels * 3
        else:
            self.spatial_down = nn.AvgPool2d(2, stride=2)
            h_channels = in_channels

        self.l_branch = LowFrequencyIlluminationBranch(
            in_channels=in_channels,
            feat_dim=feat_dim,
            num_blocks=num_l_blocks,
            use_mscb=use_mscb
        )
        self.h_branch = HighFrequencyDetailBranch(
            in_channels=h_channels,
            feat_dim=feat_dim,
            num_blocks=num_h_blocks,
            use_gdb=use_gdb
        )

        if self.use_cffm:
            self.cffm = CrossFrequencyAttentionFusion(
                low_dim=feat_dim,
                high_dim=feat_dim,
                out_dim=feat_dim
            )
        else:
            self.cffm = nn.Sequential(
                nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.fused_to_ll = nn.Conv2d(feat_dim, in_channels, kernel_size=3, padding=1)
        self.fused_to_high = nn.Conv2d(feat_dim, h_channels, kernel_size=3, padding=1)
        nn.init.normal_(self.fused_to_ll.weight, std=0.01)
        nn.init.zeros_(self.fused_to_ll.bias)
        nn.init.normal_(self.fused_to_high.weight, std=0.01)
        nn.init.zeros_(self.fused_to_high.bias)

        self.spatial_refine = nn.Sequential(
            nn.Conv2d(in_channels + feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_dim, in_channels, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.spatial_refine[-1].weight, std=0.01)
        nn.init.zeros_(self.spatial_refine[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]

        if self.use_wavelet:
            ll, high = self.dwt(x)
        else:
            ll = self.spatial_down(x)
            high = F.interpolate(x, size=(ll.shape[-2], ll.shape[-1]), mode="bilinear", align_corners=False)

        ll_rec, f_low, gain = self.l_branch(ll)
        high_rec, f_high = self.h_branch(high, gain)

        if self.use_cffm:
            f_fused = self.cffm(f_low, f_high)
        else:
            f_fused = self.cffm(torch.cat([f_low, f_high], dim=1))

        ll_rec = ll_rec + self.fused_to_ll(f_fused) * 0.2
        high_rec = high_rec + self.fused_to_high(f_fused) * 0.1

        if self.use_wavelet:
            synth_img = self.idwt(ll_rec, high_rec, orig_size=(orig_h, orig_w))
        else:
            ll_up = F.interpolate(ll_rec, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            high_up = F.interpolate(high_rec, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            synth_img = ll_up + high_up * 0.2

        f_fused_up = F.interpolate(f_fused, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        spatial_in = torch.cat([synth_img, f_fused_up], dim=1)
        out = synth_img + self.spatial_refine(spatial_in) * 0.3

        return torch.clamp(out, 0.0, 1.0)


def create_model(feat_dim: int = 64, **kwargs) -> FDRetinexNet:
    return FDRetinexNet(feat_dim=feat_dim, **kwargs)


if __name__ == "__main__":
    dummy = torch.randn(2, 3, 256, 256)
    model = create_model()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {num_params / 1e6:.3f}M")
    out = model(dummy)
    print(f"Input Shape: {dummy.shape} -> Output Shape: {out.shape}")
