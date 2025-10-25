"""
dataset.py — Dataset utilities for 3D prostate MRI segmentation using PyTorch and NIfTI files.

This script provides:
1. Dataset pairing for NIfTI image–label volumes (e.g., HipMRI or medical datasets)
2. Preprocessing (normalization, resizing, augmentation)
3. PyTorch-compatible Dataset and DataLoader generation for 3D U-Net training

Author: Benjamin Rose
Date: 2025
Project: 3D Improved U-Net (Prostate MRI Segmentation)
"""

import os
import glob
import numpy as np
import nibabel as nib
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.ndimage import zoom
import random
import logging

# -------------------------------------------------------------------------
# Logging configuration — saves info messages to logs/dataset.log
# -------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/dataset.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


# -------------------------------------------------------------------------
# Helper functions for filename handling, pairing, and resizing
# -------------------------------------------------------------------------
def _stem(p):
    """
    Return a cleaned base name (stem) from a file path.
    Removes '.nii', '.nii.gz', and dataset-specific suffixes like '_LFOV' or '_SEMANTIC'.

    Args:
        p (str): Full file path.

    Returns:
        str: Cleaned filename without extension or suffixes.
    """
    b = os.path.basename(p)
    if b.endswith('.nii.gz'):
        b = b[:-7]
    elif b.endswith('.nii'):
        b = b[:-4]
    b = b.replace('_LFOV', '').replace('_SEMANTIC', '')
    return b


def _pair_images_labels(image_dir, label_dir) -> List[Tuple[str, str]]:
    """
    Match each image file with its corresponding label file using filename stems.

    Args:
        image_dir (str): Directory containing input image volumes (.nii or .nii.gz)
        label_dir (str): Directory containing label/segmentation masks

    Returns:
        List[Tuple[str, str]]: List of (image_path, label_path) pairs
    """
    imgs = sorted(glob.glob(os.path.join(image_dir, '*.nii')) + glob.glob(os.path.join(image_dir, '*.nii.gz')))
    labs = sorted(glob.glob(os.path.join(label_dir, '*.nii')) + glob.glob(os.path.join(label_dir, '*.nii.gz')))

    # Create a dictionary mapping label stems to paths
    lab_map = {_stem(p): p for p in labs}
    pairs = []

    for ip in imgs:
        k = _stem(ip)
        if k in lab_map:
            pairs.append((ip, lab_map[k]))

    logging.info(f"Paired {len(pairs)} images with labels from {len(imgs)} images and {len(labs)} labels.")
    return pairs


def _resize_vol(vol, out_shape, order):
    """
    Resize a 3D volume to a new spatial shape using interpolation.

    Args:
        vol (np.ndarray): 3D input volume (Z, Y, X)
        out_shape (tuple): Target shape (Z, Y, X)
        order (int): Interpolation order (0=nearest, 1=linear, etc.)

    Returns:
        np.ndarray: Resized 3D volume.
    """
    in_z, in_y, in_x = vol.shape
    oz, oy, ox = out_shape
    factors = (oz / in_z, oy / in_y, ox / in_x)
    return zoom(vol, factors, order=order)


# -------------------------------------------------------------------------
# Dataset class for 3D MRI segmentation
# -------------------------------------------------------------------------
class HipMRI3DDataset(Dataset):
    """
    PyTorch Dataset for loading 3D medical image volumes (NIfTI format).

    Each item includes:
        - Normalized image volume (float32)
        - Segmentation label volume (int16)
        - Original file paths (for reference)

    Features:
        - Z-score normalization
        - Optional random flipping and rotation for augmentation
        - Resampling to a uniform spatial size

    Args:
        image_dir (str): Directory containing MRI volumes (.nii/.nii.gz)
        label_dir (str): Directory containing segmentation masks
        spatial_size (tuple): Target shape (H, W, D)
        augment (bool): Whether to apply random flips/rotations
        seed (int): Random seed for reproducibility
    """

    def __init__(self, image_dir, label_dir, spatial_size=(128, 128, 64), augment=False, seed=42):
        super().__init__()
        self.items = _pair_images_labels(image_dir, label_dir)
        self.spatial_size = spatial_size
        self.augment = augment
        random.seed(seed)
        logging.info(f"Dataset initialized with {len(self.items)} paired samples. Augment={self.augment}")

    def __len__(self):
        """Return number of paired samples."""
        return len(self.items)

    def _load(self, img_p, lab_p):
        """Load image and label volumes from disk using nibabel."""
        img = nib.load(img_p).get_fdata().astype(np.float32)
        lab = nib.load(lab_p).get_fdata().astype(np.int16)
        return img, lab

    def _norm(self, img):
        """Apply z-score normalization: (x - mean) / std."""
        m = img.mean()
        s = img.std() + 1e-8
        return (img - m) / s

    def _rand_flip(self, img, lab):
        """Randomly flip along each spatial axis with 50% probability."""
        for axis in [0, 1, 2]:
            if random.random() < 0.5:
                img = np.flip(img, axis=axis).copy()
                lab = np.flip(lab, axis=axis).copy()
        return img, lab

    def _rand_rotate90(self, img, lab):
        """Randomly rotate 90° around a random pair of axes."""
        k = random.randint(0, 3) if random.random() < 0.5 else 0
        if k:
            axes = random.choice([(0, 1), (1, 2), (0, 2)])
            img = np.rot90(img, k=k, axes=axes).copy()
            lab = np.rot90(lab, k=k, axes=axes).copy()
        return img, lab

    def __getitem__(self, idx):
        """
        Load, normalize, resize, and optionally augment a single sample.

        Returns:
            dict: {
                'image': Tensor [1, D, H, W],
                'label': Tensor [D, H, W],
                'image_path': str,
                'label_path': str
            }
        """
        img_p, lab_p = self.items[idx]
        img, lab = self._load(img_p, lab_p)

        # Normalization
        img = self._norm(img)

        # Resize to target spatial dimensions
        img = _resize_vol(img, self.spatial_size[::-1], order=1)
        lab = _resize_vol(lab, self.spatial_size[::-1], order=0)

        # Optional augmentations
        if self.augment:
            img, lab = self._rand_flip(img, lab)
            img, lab = self._rand_rotate90(img, lab)

        # Add channel dimension and convert to tensors
        img = np.expand_dims(img, 0)
        img_t = torch.from_numpy(img.copy()).float()
        lab_t = torch.from_numpy(lab.copy()).long()

        return {"image": img_t, "label": lab_t, "image_path": img_p, "label_path": lab_p}


# -------------------------------------------------------------------------
# DataLoader construction helper
# -------------------------------------------------------------------------
def make_loaders(image_dir,
                 label_dir,
                 spatial_size=(128, 128, 64),
                 batch_size=1,
                 num_workers=4,
                 val_split=0.1,
                 test_split=0.1,
                 seed=42,
                 augment_train=True):
    """
    Create PyTorch DataLoaders for training, validation, and testing splits.

    Automatically:
        - Pairs image/label volumes
        - Splits dataset into train/val/test
        - Applies augmentation only to training data
        - Ensures reproducible random splits

    Args:
        image_dir (str): Directory with input MRI volumes
        label_dir (str): Directory with segmentation masks
        spatial_size (tuple): Desired volume shape
        batch_size (int): Batch size for training
        num_workers (int): DataLoader worker threads
        val_split (float): Fraction of dataset for validation
        test_split (float): Fraction for testing
        seed (int): Random seed for reproducibility
        augment_train (bool): Whether to augment training data

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create base dataset
    full = HipMRI3DDataset(image_dir, label_dir, spatial_size=spatial_size, augment=augment_train, seed=seed)
    n = len(full)

    # Determine split sizes
    n_test = int(round(n * test_split))
    n_val = int(round(n * val_split))
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)

    logging.info(f"Dataset split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Split into subsets
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=gen)

    # Disable augmentation for val/test sets
    val_ds.dataset.augment = False
    test_ds.dataset.augment = False

    # Construct DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    logging.info("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader
