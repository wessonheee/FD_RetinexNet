import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PairedLowLightDataset(Dataset):
    def __init__(
        self,
        low_dir: str,
        high_dir: str,
        patch_size: int = 256,
        is_train: bool = True,
    ):
        super().__init__()
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.patch_size = patch_size
        self.is_train = is_train

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.filenames = [
            f for f in os.listdir(low_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
            and os.path.exists(os.path.join(high_dir, f))
        ]
        self.filenames.sort()

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No paired images found between {low_dir} and {high_dir}! Please check file paths."
            )

    def __len__(self):
        return len(self.filenames)

    def _augment(self, low_img: Image.Image, high_img: Image.Image):
        # Random crop
        w, h = low_img.size
        th, tw = self.patch_size, self.patch_size
        if w > tw and h > th:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            low_img = TF.crop(low_img, i, j, th, tw)
            high_img = TF.crop(high_img, i, j, th, tw)
        elif w != tw or h != th:
            low_img = TF.resize(low_img, (th, tw))
            high_img = TF.resize(high_img, (th, tw))

        # Random horizontal flipping
        if random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)

        # Random vertical flipping
        if random.random() > 0.5:
            low_img = TF.vflip(low_img)
            high_img = TF.vflip(high_img)

        # Random 90 deg rotation
        rot_choice = random.choice([0, 90, 180, 270])
        if rot_choice > 0:
            low_img = TF.rotate(low_img, rot_choice)
            high_img = TF.rotate(high_img, rot_choice)

        return low_img, high_img

    def __getitem__(self, index: int):
        fname = self.filenames[index]
        low_path = os.path.join(self.low_dir, fname)
        high_path = os.path.join(self.high_dir, fname)

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if self.is_train:
            low_img, high_img = self._augment(low_img, high_img)

        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        return {
            "low": low_tensor,
            "high": high_tensor,
            "name": fname
        }
