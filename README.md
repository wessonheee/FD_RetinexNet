<div align="center">

# FD-RetinexNet: Frequency-Decoupled Retinex Network for Ultra-Lightweight Low-Light Image Enhancement


<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?style=flat&logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?style=flat&logo=python" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://github.com/wessonheee/FD_RetinexNet"><img src="https://img.shields.io/badge/GitHub-FD__RetinexNet-181717.svg?logo=github" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/Edge%20Throughput-125%20FPS-success.svg" alt="FPS">
  <img src="https://img.shields.io/badge/Parameters-0.849M-brightgreen.svg" alt="Params">
</p>

---

<p align="center">
  <strong>Official PyTorch Implementation of FD-RetinexNet</strong><br>
  <em>A physically grounded, frequency-decoupled Retinex architecture tailored for real-time edge low-light image enhancement.</em>
</p>
## ✨ Key Highlights

- **Orthogonal Wavelet Frequency Decoupling**: Employs a 2D Haar Discrete Wavelet Transform (DWT) to losslessly divide the latent space into an approximation subband ($LL$) and three directional detail subbands ($LH, HL, HH$), physically terminating the spatial noise-illumination entanglement.
- **Dedicated Micro-Architectural Synergy**:
  - **MSCB (Multi-Scale Context Block)**: Receptive field expansion with multi-dilation depthwise paths ($d \in \{1, 2, 4\}$) for smooth, halo-free global illumination adjustment without noise surge.
  - **GDB (Gated Detail Block)**: Spatial gating mechanism designed to suppress dark-current sensor noise while selectively preserving structural high-frequency edge contours.
  - **CFFM (Cross-Frequency Fusion Module)**: Bidirectional edge-guided channel attention and illumination-guided spatial attention ensuring cross-spectral alignment before Inverse DWT synthesis.
- **Ultra-Lightweight & Pareto-Superior**:
  - **0.849M Parameters** and **18.5 GFLOPs** (30.7$\times$ fewer parameters and 7.6$\times$ fewer FLOPs than Restormer).
  - Real-time throughput of **125 FPS** (8.0 ms/frame) on an edge GPU platform.
- **Multi-Objective Compound Loss**: Jointly regularizes pixel-level smooth Charbonnier fidelity, structural similarity ($\mathcal{L}_{\text{SSIM}}$), Fourier frequency amplitude consistency ($\mathcal{L}_{\text{freq}}$), and angular color constancy ($\mathcal{L}_{\text{color}}$).

---

## 🖼️ Visual Motivation & Architecture

### 1. The Noise-Illumination Dilemma
In conventional spatial Retinex pipelines, boosting illumination inevitably amplifies high-frequency stochastic sensor noise. FD-RetinexNet achieves clean noise suppression and natural illumination recovery by operating strictly within orthogonal frequency subbands.


### 2. Core Micro-Modules (MSCB, GDB, CFFM)

---

## 📊 Benchmark Performance

### Quantitative Comparison on LOL-v1 Benchmark
FD-RetinexNet achieves state-of-the-art trade-offs between restoration fidelity (PSNR / SSIM) and edge computing efficiency (Parameters, FLOPs, Latency, FPS):

| Method | Venue & Year | Model Paradigm | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | Params (M) $\downarrow$ | FLOPs (G) $\downarrow$ | Latency (ms) $\downarrow$ | FPS $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RetinexNet | BMVC'18 | Deep Retinex | 16.77 | 0.562 | 0.388 | 0.55 | 587.5 | 85.0 | 11.8 |
| Zero-DCE | CVPR'20 | Zero-Ref Curve | 14.86 | 0.559 | 0.335 | 0.08 | 5.2 | 3.2 | 312.5 |
| EnlightenGAN | TIP'21 | Unpaired GAN | 17.48 | 0.658 | 0.322 | 8.64 | 245.0 | 55.6 | 18.0 |
| KinD | IJCV'21 | Deep Retinex | 20.87 | 0.805 | 0.178 | 8.02 | 120.4 | 32.2 | 31.0 |
| SCI | CVPR'22 | Self-Calibrated | 14.78 | 0.525 | 0.360 | **0.0003** | **4.1** | **2.9** | **344.8** |
| URetinex-Net | CVPR'22 | Deep Unfolding | 21.33 | 0.835 | 0.121 | 0.36 | 45.3 | 16.1 | 62.1 |
| Restormer | CVPR'22 | Transformer | 22.43 | 0.824 | 0.147 | 26.10 | 141.0 | 78.1 | 12.8 |
| PairLIE | CVPR'23 | Prior Learning | 21.75 | 0.822 | 0.134 | 0.45 | 38.5 | 18.5 | 54.0 |
| FourLLIE | ACM MM'23 | Fourier Freq. | 23.49 | 0.841 | 0.119 | 2.15 | 32.6 | 22.2 | 45.0 |
| Retinexformer | ICCV'23 | Retinex Trans. | 25.26 | 0.849 | 0.118 | 1.59 | 54.2 | 38.5 | 26.0 |
| CSDNet | TMM'24 | Color Decouple | 23.98 | 0.845 | 0.120 | 1.82 | 41.0 | 25.0 | 40.0 |
| HVI-CIDNet | CVPR'24 | Color Decouple | **25.85** | **0.865** | **0.109** | 0.98 | 26.4 | 14.5 | 69.0 |
| **FD-RetinexNet (Ours)** | **This Work** | **Wavelet Retinex** | **22.70** | **0.846** | **0.125** | **0.85** | **18.5** | **8.0** | **125.0** |

> **Note**: In challenging extreme low-light scenes (e.g., Bedroom `22.png`, input 6.11 dB), FD-RetinexNet achieves up to **28.65 dB PSNR** and **0.899 SSIM**, completely eliminating sensor color noise while preserving fine knitted textures.

## 

## 🛠️ Installation & Environment

### 1. Clone the repository
```bash
git clone https://github.com/wessonheee/FD_RetinexNet.git
cd FD_RetinexNet
```

### 2. Create and activate Conda environment
```bash
conda create -n fd_retinex python=3.8 -y
conda activate fd_retinex
```

### 3. Install PyTorch and dependencies
Install PyTorch matching your CUDA version (e.g., CUDA 11.8 / 12.1):
```bash
# Example for CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install numpy Pillow opencv-python lpips scikit-image tqdm matplotlib
```

---

## 📦 Dataset Preparation

Download the standard **LOL-v1 benchmark dataset** from [official website](https://daoosrn.github.io/dataset/lol.html) or Google Drive.

---

## 🚀 Quick Start (Inference & Demo)

### 1. Batch Image Enhancement
Run inference on a folder of low-light images using our pre-trained model checkpoint:
```bash
python test_and_visualize.py \
    --weights checkpoints/best_model.pth \
    --input_dir datasets/LOLv1/eval15/low \
    --output_dir results/enhanced/ \
    --device cuda
```

---

## 🏋️ Training from Scratch

To train **FD-RetinexNet** from scratch :

```bash
python train.py \
    --train_low datasets/LOLv1/our485/low \
    --train_high datasets/LOLv1/our485/high \
    --val_low datasets/LOLv1/eval15/low \
    --val_high datasets/LOLv1/eval15/high \
    --checkpoint_dir checkpoints/ \
    --epochs 200 \
    --batch_size 8 \
    --patch_size 256 \
    --lr 3e-4 \
    --save_freq 20
```

- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$)
- **Learning Rate Schedule**: Cosine Annealing decay ($3.0 \times 10^{-4} \to 1.0 \times 10^{-6}$)
- **FP32 Precision**: Retains smooth numerical frequency gradient stability during wavelet decomposition.

---

## 📄 License & Acknowledgments

- This repository is released under the **Apache 2.0 License**. See [LICENSE](LICENSE) for more details.
- We gratefully acknowledge the creators of the open-source contributions from the low-light enhancement community.

