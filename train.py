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
from losses.compound_loss import CompoundEnhancementLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train FD-RetinexNet on LOL Dataset")
    parser.add_argument("--train_low", type=str, required=True, help="Path to train low-light folder")
    parser.add_argument("--train_high", type=str, required=True, help="Path to train normal-light folder")
    parser.add_argument("--val_low", type=str, default=None, help="Path to test low-light folder")
    parser.add_argument("--val_high", type=str, default=None, help="Path to test normal-light folder")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (8 is ideal for T4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--feat_dim", type=int, default=64, help="Feature dimension (default: 64, 0.849M params)")
    parser.add_argument("--patch_size", type=int, default=256, help="Training patch crop size")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency in epochs")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP mixed precision (default: False for numerical stability in frequency restoration)")
    return parser.parse_args()


def calc_batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR for a batch of tensors in [0, 1]."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
    return float(torch.mean(psnr).item())


def get_scaler(device, use_amp=False):
    if not use_amp or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        try:
            return torch.cuda.amp.GradScaler()
        except Exception:
            return None


def autocast_context(device):
    if device.type == "cuda":
        try:
            return torch.amp.autocast("cuda")
        except Exception:
            try:
                return torch.cuda.amp.autocast()
            except Exception:
                pass
    import contextlib
    return contextlib.nullcontext()


def validate(model, val_loader, device):
    model.eval()
    psnr_vals = []
    with torch.no_grad():
        for batch in val_loader:
            low = batch["low"].to(device)
            high = batch["high"].to(device)
            with autocast_context(device):
                pred = model(low)
            psnr_vals.append(calc_batch_psnr(pred, high))
    mean_psnr = np.mean(psnr_vals) if psnr_vals else 0.0
    return mean_psnr


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("Starting FD-RetinexNet Training Pipeline")
    print(f"Device: {device} | Total Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 60)

    # 1. Dataset & DataLoader
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
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    if args.val_low and args.val_high:
        val_dataset = PairedLowLightDataset(
            low_dir=args.val_low,
            high_dir=args.val_high,
            patch_size=args.patch_size,
            is_train=False
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 2. Model, Criterion, Optimizer, Scheduler, Scaler
    model = FDRetinexNet(feat_dim=args.feat_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FD-RetinexNet Parameters: {num_params / 1e6:.3f}M", flush=True)
    criterion = CompoundEnhancementLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = get_scaler(device, use_amp=args.use_amp)
    mode_str = "AMP Mixed Precision" if scaler is not None else "Pure FP32 (Full Precision)"
    print(f"Precision Mode: {mode_str}", flush=True)

    start_epoch = 1
    best_psnr = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming checkpoint from {args.resume}...", flush=True)
        try:
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed successfully at epoch {start_epoch} (Best PSNR: {best_psnr:.2f} dB)", flush=True)

    # 3. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)

            optimizer.zero_grad()
            if scaler is not None:
                with autocast_context(device):
                    pred = model(low)
                    loss, loss_dict = criterion(pred, high)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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

        # Validation
        val_info = ""
        if val_loader is not None and (epoch % 5 == 0 or epoch == args.epochs):
            val_psnr = validate(model, val_loader, device)
            val_info = f" | Val PSNR: {val_psnr:.2f} dB"

            if val_psnr > best_psnr:
                best_psnr = float(val_psnr)
                best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save({"model": model.state_dict(), "best_psnr": float(best_psnr), "epoch": int(epoch)}, best_path)
                val_info += " [NEW BEST! Saved]"

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {epoch_loss:.4f}{val_info} | LR: {curr_lr:.2e} | Time: {duration:.1f}s", flush=True)

        # Regular periodic save
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            last_path = os.path.join(args.checkpoint_dir, "last_model.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": int(epoch),
                "best_psnr": float(best_psnr)
            }, last_path)

    print("=" * 60, flush=True)
    print(f"Training Complete! Best Validation PSNR: {best_psnr:.2f} dB", flush=True)
    print(f"Best model saved to {os.path.join(args.checkpoint_dir, 'best_model.pth')}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
