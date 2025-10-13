# dataset.py
import os
import glob
import numpy as np
import nibabel as nib
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import zoom
import random

def _stem(p):
    b = os.path.basename(p)
    if b.endswith('.nii.gz'):
        return b[:-7]
    if b.endswith('.nii'):
        return b[:-4]
    return os.path.splitext(b)[0]

def _pair_images_labels(image_dir, label_dir) -> List[Tuple[str, str]]:
    imgs = sorted(glob.glob(os.path.join(image_dir, '*.nii')) + glob.glob(os.path.join(image_dir, '*.nii.gz')))
    labs = sorted(glob.glob(os.path.join(label_dir, '*.nii')) + glob.glob(os.path.join(label_dir, '*.nii.gz')))
    lab_map = {_stem(p): p for p in labs}
    pairs = []
    for ip in imgs:
        k = _stem(ip)
        if k in lab_map:
            pairs.append((ip, lab_map[k]))
    return pairs

def _resize_vol(vol, out_shape, order):
    in_z, in_y, in_x = vol.shape
    oz, oy, ox = out_shape
    factors = (oz / in_z, oy / in_y, ox / in_x)
    return zoom(vol, factors, order=order)

class HipMRI3DDataset(Dataset):
    def __init__(self, image_dir, label_dir, spatial_size=(128,128,64), augment=False, seed=42):
        super().__init__()
        self.items = _pair_images_labels(image_dir, label_dir)
        self.spatial_size = spatial_size
        self.augment = augment
        random.seed(seed)

    def __len__(self):
        return len(self.items)

    def _load(self, img_p, lab_p):
        img = nib.load(img_p).get_fdata().astype(np.float32)
        lab = nib.load(lab_p).get_fdata().astype(np.int16)
        return img, lab

    def _norm(self, img):
        m = img.mean()
        s = img.std() + 1e-8
        return (img - m) / s

    def _rand_flip(self, img, lab):
        for axis in [0,1,2]:
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                lab = np.flip(lab, axis=axis).copy()
        return img, lab

    def _rand_rotate90(self, img, lab):
        k = random.randint(0,3) if random.random() < 0.5 else 0
        if k:
            axes = random.choice([(0,1),(1,2),(0,2)])
            img = np.rot90(img, k=k, axes=axes).copy()
            lab = np.rot90(lab, k=k, axes=axes).copy()
        return img, lab

    def __getitem__(self, idx):
        img_p, lab_p = self.items[idx]
        img, lab = self._load(img_p, lab_p)
        img = self._norm(img)
        img = _resize_vol(img, self.spatial_size[::-1], order=1)
        lab = _resize_vol(lab, self.spatial_size[::-1], order=0)
        if self.augment:
            img, lab = self._rand_flip(img, lab)
            img, lab = self._rand_rotate90(img, lab)
        img = np.expand_dims(img, 0)
        img_t = torch.from_numpy(img.copy()).float()
        lab_t = torch.from_numpy(lab.copy()).long()
        return {"image": img_t, "label": lab_t, "image_path": img_p, "label_path": lab_p}

def make_loaders(image_dir,
                 label_dir,
                 spatial_size=(128,128,64),
                 batch_size=1,
                 num_workers=4,
                 val_split=0.1,
                 test_split=0.1,
                 seed=42,
                 augment_train=True):
    full = HipMRI3DDataset(image_dir, label_dir, spatial_size=spatial_size, augment=augment_train, seed=seed)
    n = len(full)
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)
    # no augmentation leakage
    val_ds.dataset.augment = False
    test_ds.dataset.augment = False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
