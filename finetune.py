import os
import argparse
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF
from PIL import Image

from models.fd_retinex import FDRetinexNet
try:
    from data_loader.lol_dataset import PairedLowLightDataset
except ImportError:
    try:
        from lol_dataset import PairedLowLightDataset
    except ImportError:
        from datasets.lol_dataset import PairedLowLightDataset
from losses.compound_loss import SSIMLoss, FrequencyAmplitudeLoss, ColorCosineLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune FD-RetinexNet for SOTA PSNR Boost")
    parser.add_argument("--train_low", type=str, required=True, help="Path to train low-light folder")
    parser.add_argument("--train_high", type=str, required=True, help="Path to train normal-light folder")
    parser.add_argument("--val_low", type=str, required=True, help="Path to val low-light folder")
    parser.add_argument("--val_high", type=str, required=True, help="Path to val normal-light folder")
    parser.add_argument("--pretrained", type=str, default="checkpoints/best_model.pth", help="Pretrained 22.70dB weights")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_finetune/", help="Directory to save boosted checkpoints")
    parser.add_argument("--epochs", type=int, default=40, help="Total finetune epochs (30-50 recommended)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Finetuning learning rate")
    parser.add_argument("--feat_dim", type=int, default=64, help="Model feature dimension")
    parser.add_argument("--patch_size", type=int, default=256, help="Training patch crop size")
    parser.add_argument("--val_freq", type=int, default=2, help="Validation frequency in epochs")
    return parser.parse_args()


class PSNROptimizedCompoundLoss(torch.nn.Module):
    """
    PSNR-Maximizing Objective:
    Directly aligns optimization with MSE while retaining perceptual stability:
      - Charbonnier (L1 robust proxy)
      - Direct MSE (2.0x weight: mathematically pushes PSNR up)
      - Mild SSIM (0.05: prevents edge blurring without hijacking pixel convergence)
      - Frequency amplitude (0.05) & Color Cosine (0.05)
    """
    def __init__(self):
        super().__init__()
        self.charbonnier_eps2 = 1e-6
        self.mse_loss = torch.nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.freq_loss = FrequencyAmplitudeLoss()
        self.color_loss = ColorCosineLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()

        # Charbonnier Loss
        diff = pred - target
        l_char = torch.mean(torch.sqrt(diff * diff + self.charbonnier_eps2))
        
        # MSE Loss (Direct PSNR Maximizer)
        l_mse = self.mse_loss(pred, target)
        
        # Perceptual & Prior Guidance (Balanced at 0.05)
        l_ssim = self.ssim_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        l_color = self.color_loss(pred, target)

        total = l_char + 2.0 * l_mse + 0.05 * l_ssim + 0.05 * l_freq + 0.05 * l_color
        return total, {
            "total": total.item(),
            "mse": l_mse.item(),
            "char": l_char.item(),
            "ssim": l_ssim.item()
        }


def rgb_to_y_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [B, 3, H, W] in [0, 1] to Y luminance channel in [0, 1]."""
    r = tensor[:, 0:1, :, :]
    g = tensor[:, 1:2, :, :]
    b = tensor[:, 2:3, :, :]
    y = (65.481 * r + 128.553 * g + 24.966 * b + 16.0) / 255.0
    return y


def calc_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Calculate both RGB PSNR and standard benchmark Y-channel PSNR."""
    # 1. RGB PSNR
    mse_rgb = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr_rgb = float(torch.mean(10.0 * torch.log10(1.0 / (mse_rgb + 1e-8))).item())

    # 2. Y-channel PSNR (Standard Benchmark Protocol)
    pred_y = rgb_to_y_tensor(pred)
    target_y = rgb_to_y_tensor(target)
    mse_y = torch.mean((pred_y - target_y) ** 2, dim=[1, 2, 3])
    psnr_y = float(torch.mean(10.0 * torch.log10(1.0 / (mse_y + 1e-8))).item())

    return psnr_rgb, psnr_y


def validate(model, val_loader, device):
    model.eval()
    rgb_list, y_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            low = batch["low"].to(device)
            high = batch["high"].to(device)
            pred = model(low)
            p_rgb, p_y = calc_metrics(pred, high)
            rgb_list.append(p_rgb)
            y_list.append(p_y)
    return float(np.mean(rgb_list)), float(np.mean(y_list))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 65)
    print("Starting FD-RetinexNet SOTA Finetuning Stage (Path B)")
    print(f"Device: {device} | Epochs: {args.epochs} | LR: {args.lr:.1e}")
    print(f"Pretrained Weights: {args.pretrained}")
    print("=" * 65)

    # 1. DataLoaders
    train_dataset = PairedLowLightDataset(
        low_dir=args.train_low,
        high_dir=args.train_high,
        patch_size=args.patch_size,
        is_train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = PairedLowLightDataset(
        low_dir=args.val_low,
        high_dir=args.val_high,
        patch_size=args.patch_size,
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. Model & Load Pretrained Weights
    model = FDRetinexNet(feat_dim=args.feat_dim).to(device)
    if os.path.exists(args.pretrained):
        print(f"--> Loading pre-trained 22.70 dB model from {args.pretrained}...")
        try:
            ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        base_psnr = ckpt.get("best_psnr", 22.70)
        print(f"--> Pretrained weights loaded successfully! Baseline PSNR: {base_psnr:.2f} dB")
    else:
        print(f"Warning: {args.pretrained} not found! Starting without pretrained checkpoint.")
        base_psnr = 0.0

    # 3. Initial Validation Check to Confirm Baseline
    init_rgb, init_y = validate(model, val_loader, device)
    print(f"--> Verified Pre-finetune Score: RGB PSNR = {init_rgb:.2f} dB | Y-channel PSNR = {init_y:.2f} dB")

    # 4. Optimizer, Scheduler & Objective
    criterion = PSNROptimizedCompoundLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)

    best_rgb_psnr = init_rgb
    best_y_psnr = init_y

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(low)
            loss, loss_dict = criterion(pred, high)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        duration = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        val_info = ""
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            rgb_val, y_val = validate(model, val_loader, device)
            val_info = f" | RGB: {rgb_val:.2f} dB | Y: {y_val:.2f} dB"
            if rgb_val > best_rgb_psnr:
                best_rgb_psnr = rgb_val
                best_y_psnr = y_val
                save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save({
                    "model": model.state_dict(),
                    "best_psnr": float(best_rgb_psnr),
                    "best_y_psnr": float(best_y_psnr),
                    "epoch": epoch
                }, save_path)
                val_info += " [NEW BEST! 🔥]"

        print(f"Finetune [{epoch:02d}/{args.epochs:02d}] | Loss: {epoch_loss:.4f}{val_info} | LR: {curr_lr:.2e} | Time: {duration:.1f}s", flush=True)

    print("=" * 65)
    print(f"Finetuning Completed!")
    print(f"Final Best RGB PSNR: {best_rgb_psnr:.2f} dB (Gain: +{best_rgb_psnr - init_rgb:.2f} dB)")
    print(f"Final Best Y-channel PSNR: {best_y_psnr:.2f} dB (Gain: +{best_y_psnr - init_y:.2f} dB)")
    print(f"Best boosted model saved to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
