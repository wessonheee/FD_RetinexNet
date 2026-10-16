import os
import argparse
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms.functional as TF

from models.fd_retinex import FDRetinexNet


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and Paper Visualizer for FD-RetinexNet")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Input low-light images directory")
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feat_dim", type=int, default=64, help="Model feature dimension")
    parser.add_argument("--zoom_box", type=int, nargs=4, default=None,
                        help="Optional zoom box: [ymin, xmin, ymax, xmax] for paper figures")
    return parser.parse_args()


def add_zoom_in_inset(
    img: Image.Image,
    box: list,
    box_color: str = "red",
    inset_scale: float = 3.0,
    corner: str = "bottom_right"
) -> Image.Image:
    """
    Draw a colored rectangle on the image and paste an enlarged inset patch in one corner.
    Ideal for academic paper visual comparison figures.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    ymin, xmin, ymax, xmax = box

    # Draw bounding box
    draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=3)

    # Crop and zoom patch
    patch = img.crop((xmin, ymin, xmax, ymax))
    pw, ph = patch.size
    new_w, new_h = int(pw * inset_scale), int(ph * inset_scale)
    patch_zoomed = patch.resize((new_w, new_h), Image.Resampling.NEAREST)

    # Border around inset
    patch_draw = ImageDraw.Draw(patch_zoomed)
    patch_draw.rectangle([0, 0, new_w - 1, new_h - 1], outline=box_color, width=3)

    # Place in corner
    w, h = img.size
    if corner == "bottom_right":
        pos = (w - new_w - 5, h - new_h - 5)
    elif corner == "top_right":
        pos = (w - new_w - 5, 5)
    elif corner == "bottom_left":
        pos = (5, h - new_h - 5)
    else:
        pos = (5, 5)

    img_copy.paste(patch_zoomed, pos)
    return img_copy


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading FD-RetinexNet from {args.weights}...")
    model = FDRetinexNet(feat_dim=args.feat_dim).to(args.device)
    if os.path.exists(args.weights):
        try:
            state = torch.load(args.weights, map_location=args.device, weights_only=False)
        except TypeError:
            state = torch.load(args.weights, map_location=args.device)
        model.load_state_dict(state.get("model", state))
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: {args.weights} not found. Running with initial weights (test mode).")

    model.eval()
    fnames = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"Found {len(fnames)} images to enhance in {args.input_dir}")

    with torch.no_grad():
        for fname in fnames:
            in_path = os.path.join(args.input_dir, fname)
            raw_img = Image.open(in_path).convert("RGB")
            tensor_in = TF.to_tensor(raw_img).unsqueeze(0).to(args.device)

            out_tensor = model(tensor_in)
            out_img = TF.to_pil_image(out_tensor.squeeze(0).cpu())

            out_path = os.path.join(args.output_dir, fname)
            out_img.save(out_path)

            if args.zoom_box is not None:
                zoomed = add_zoom_in_inset(out_img, args.zoom_box)
                base, ext = os.path.splitext(fname)
                zoom_path = os.path.join(args.output_dir, f"{base}_zoomed{ext}")
                zoomed.save(zoom_path)

    print(f"Enhancement complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
