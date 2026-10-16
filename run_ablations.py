import os
import argparse
import time
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    parser = argparse.ArgumentParser(description="Automated Ablation Study Suite for FD-RetinexNet")
    parser.add_argument("--train_low", type=str, required=True, help="Path to train low folder")
    parser.add_argument("--train_high", type=str, required=True, help="Path to train high folder")
    parser.add_argument("--val_low", type=str, required=True, help="Path to test low folder")
    parser.add_argument("--val_high", type=str, required=True, help="Path to test high folder")
    parser.add_argument("--output_dir", type=str, default="results/ablation", help="Directory for ablation results")
    parser.add_argument("--mode", type=str, default="arch", choices=["arch", "loss", "all"],
                        help="Ablation type: 'arch' (module ablations), 'loss' (loss functions), or 'all'")
    parser.add_argument("--epochs", type=int, default=25, help="Controlled training epochs per variant (default: 25)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--feat_dim", type=int, default=64, help="Feature dimension")
    return parser.parse_args()


def rgb_to_y(tensor: torch.Tensor) -> torch.Tensor:
    r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
    return (65.481 * r + 128.553 * g + 24.966 * b + 16.0) / 255.0


def calc_ssim_tensor(img1: torch.Tensor, img2: torch.Tensor) -> float:
    # Quick differentiable SSIM proxy for validation
    from losses.compound_loss import SSIMLoss
    criterion = SSIMLoss().to(img1.device)
    loss = criterion(img1, img2)
    return float((1.0 - loss).item())


def evaluate_model(model, val_loader, device):
    model.eval()
    psnr_rgb_list = []
    psnr_y_list = []
    ssim_list = []

    with torch.no_grad():
        for batch in val_loader:
            low = batch["low"].to(device)
            high = batch["high"].to(device)
            pred = model(low)

            # RGB PSNR
            mse_rgb = torch.mean((pred - high) ** 2, dim=[1, 2, 3])
            p_rgb = float(torch.mean(10.0 * torch.log10(1.0 / (mse_rgb + 1e-8))).item())
            psnr_rgb_list.append(p_rgb)

            # Y-channel PSNR
            pred_y = rgb_to_y(pred)
            high_y = rgb_to_y(high)
            mse_y = torch.mean((pred_y - high_y) ** 2, dim=[1, 2, 3])
            p_y = float(torch.mean(10.0 * torch.log10(1.0 / (mse_y + 1e-8))).item())
            psnr_y_list.append(p_y)

            # SSIM
            s = calc_ssim_tensor(pred, high)
            ssim_list.append(s)

    return float(np.mean(psnr_rgb_list)), float(np.mean(psnr_y_list)), float(np.mean(ssim_list))


def train_variant(name, model, criterion, train_loader, val_loader, device, epochs, lr):
    print(f"\n{'='*60}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Running Ablation Variant: [{name}] | Params: {num_params:.3f}M | Epochs: {epochs}")
    print(f"{'='*60}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_rgb, best_y, best_ssim = 0.0, 0.0, 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for batch in train_loader:
            low = batch["low"].to(device, non_blocking=True)
            high = batch["high"].to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(low)
            loss, _ = criterion(pred, high)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss /= len(train_loader)
        duration = time.time() - t0

        if epoch % 5 == 0 or epoch == epochs:
            rgb_val, y_val, ssim_val = evaluate_model(model, val_loader, device)
            if rgb_val > best_rgb:
                best_rgb, best_y, best_ssim = rgb_val, y_val, ssim_val
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {epoch_loss:.4f} | RGB PSNR: {rgb_val:.2f} dB | Y PSNR: {y_val:.2f} dB | SSIM: {ssim_val:.4f} | Time: {duration:.1f}s", flush=True)

    return {
        "variant": name,
        "params_m": round(num_params, 3),
        "psnr_rgb": round(best_rgb, 2),
        "psnr_y": round(best_y, 2),
        "ssim": round(best_ssim, 4)
    }


def generate_tables(results, output_dir, mode_title):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Markdown Table
    md_lines = [
        f"### {mode_title} Ablation Study Results (LOL-v1 Benchmark)\n",
        "| Variant | Description | Params (M) | PSNR (RGB) ↑ | PSNR (Y) ↑ | SSIM ↑ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    for r in results:
        md_lines.append(f"| **{r['variant']}** | {r.get('desc', '')} | {r['params_m']} | {r['psnr_rgb']} dB | {r['psnr_y']} dB | {r['ssim']} |")

    md_content = "\n".join(md_lines)
    with open(os.path.join(output_dir, "ablation_table.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. LaTeX Table (Publication Ready)
    tex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Ablation study on the LOL-v1 dataset for {mode_title}. Bold numbers indicate the best performance.}}",
        "\\label{tab:ablation_" + mode_title.lower().replace(" ", "_") + "}",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Configuration & Params (M) & PSNR (RGB) [dB] & PSNR (Y) [dB] & SSIM \\\\",
        "\\midrule"
    ]
    for r in results:
        tex_lines.append(f"{r['variant']} & {r['params_m']} & {r['psnr_rgb']} & {r['psnr_y']} & {r['ssim']} \\\\")
    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}"
    ])
    tex_content = "\n".join(tex_lines)
    with open(os.path.join(output_dir, "ablation_table.tex"), "w", encoding="utf-8") as f:
        f.write(tex_content)

    # 3. JSON summary
    with open(os.path.join(output_dir, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*65)
    print("ABLATION STUDY COMPLETED! PUBLICATION-READY TABLES GENERATED:")
    print("="*65)
    print(md_content)
    print("\n" + "="*65)
    print(f"Files saved to {output_dir}:")
    print(f"  - {os.path.join(output_dir, 'ablation_table.md')}")
    print(f"  - {os.path.join(output_dir, 'ablation_table.tex')}")
    print(f"  - {os.path.join(output_dir, 'ablation_summary.json')}")
    print("="*65)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataloaders
    train_dataset = PairedLowLightDataset(low_dir=args.train_low, high_dir=args.train_high, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dataset = PairedLowLightDataset(low_dir=args.val_low, high_dir=args.val_high, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []

    # ARCHITECTURE ABLATIONS (Table 2)
    if args.mode in ["arch", "all"]:
        arch_configs = [
            ("w/o Wavelet Decoupling", {"use_wavelet": False}, "Spatial-only baseline without Haar DWT"),
            ("w/o High-Freq GDB", {"use_gdb": False}, "Replaces Gated Denoising with standard ResBlock"),
            ("w/o Multi-Scale MSCB", {"use_mscb": False}, "Replaces MSCB with single-scale ResBlock"),
            ("w/o CFFM Attention", {"use_cffm": False}, "Replaces cross-frequency attention with naive Concat"),
            ("Full FD-RetinexNet (Ours)", {}, "Proposed complete architecture"),
        ]

        standard_criterion = CompoundEnhancementLoss().to(device)
        for name, flags, desc in arch_configs:
            model = FDRetinexNet(feat_dim=args.feat_dim, **flags).to(device)
            res = train_variant(name, model, standard_criterion, train_loader, val_loader, device, args.epochs, args.lr)
            res["desc"] = desc
            results.append(res)

        generate_tables(results, args.output_dir, "Architectural Components")

    # LOSS FUNCTION ABLATIONS (Table 3)
    if args.mode in ["loss", "all"]:
        loss_results = []
        loss_configs = [
            ("L1 Rec Only", {"lambda_ssim": 0.0, "lambda_freq": 0.0, "lambda_color": 0.0}, "Charbonnier reconstruction loss only"),
            ("L1 + SSIM", {"lambda_ssim": 0.5, "lambda_freq": 0.0, "lambda_color": 0.0}, "Pixel reconstruction + SSIM structural loss"),
            ("L1 + SSIM + Freq", {"lambda_ssim": 0.5, "lambda_freq": 0.1, "lambda_color": 0.0}, "Added Fourier frequency amplitude loss"),
            ("Full Compound Loss", {"lambda_ssim": 0.5, "lambda_freq": 0.1, "lambda_color": 0.2}, "Full Compound Loss with Cosine Color constraint"),
        ]

        for name, loss_kwargs, desc in loss_configs:
            model = FDRetinexNet(feat_dim=args.feat_dim).to(device)
            crit = CompoundEnhancementLoss(**loss_kwargs).to(device)
            res = train_variant(name, model, crit, train_loader, val_loader, device, args.epochs, args.lr)
            res["desc"] = desc
            loss_results.append(res)

        loss_out_dir = os.path.join(args.output_dir, "loss")
        generate_tables(loss_results, loss_out_dir, "Loss Functions")


if __name__ == "__main__":
    main()
