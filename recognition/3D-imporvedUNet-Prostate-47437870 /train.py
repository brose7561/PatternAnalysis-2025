# © Benjamin Rose, 2025
# please see attached licence - MIT
# Description: 3D MRI segmentation training pipeline using Improved UNet3D.
# Includes dataloading, mixed-precision training, early stopping, LR scheduling,
# checkpoint saving, and Dice evaluation on the test split.

import os
import argparse
import math
import logging
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from modules import UNet3DImproved
from dataset import make_loaders
from utils import to_device, per_class_dice_from_logits, ce_dice_loss, save_curves

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)


class EarlyStop:
    """Track validation loss and stop training once it stops improving."""
    def __init__(self, patience=25, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.count = 0
        self.should_stop = False

    def step(self, val_loss):
        """Update counter and flag if validation loss plateaus."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.should_stop = True


def parse_args():
    """Command-line configuration for training."""
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', type=str, default='/home/groups/comp3710/HipMRI_Study_open/semantic_MRs')
    p.add_argument('--label_dir', type=str, default='/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only')
    p.add_argument('--spatial_size', type=int, nargs=3, default=[128, 128, 64])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base_ch', type=int, default=16)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--ignore_index', type=int, default=0)
    p.add_argument('--outdir', type=str, default='runs_improved_unet3d')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--patience', type=int, default=25)
    return p.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    device = to_device()
    os.makedirs(args.outdir, exist_ok=True)

    # Prepare loaders for train, validation and test sets
    train_loader, val_loader, test_loader = make_loaders(
        args.image_dir, args.label_dir,
        spatial_size=tuple(args.spatial_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1, test_split=0.1,
        seed=args.seed,
        augment_train=args.augment
    )

    # Initialize model, optimizer and learning-rate scheduler
    model = UNet3DImproved(
        in_channels=1,
        num_classes=args.num_classes,
        base_ch=args.base_ch,
        depth=args.depth,
        dropout_p=args.dropout
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Configure mixed precision when supported
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        amp_ctx = torch.cuda.amp.autocast()
    elif hasattr(torch, "amp") and device.type == 'mps':
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        amp_ctx = torch.amp.autocast('mps')  # MPS: no dynamic scaling
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        amp_ctx = nullcontext()

    early = EarlyStop(patience=args.patience)
    best_val = math.inf
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    logging.info(f"Training for {args.epochs} epochs on {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')

        for batch in pbar:
            imgs = batch['image'].to(device, non_blocking=True)
            labs = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = model(imgs)
                loss = ce_dice_loss(
                    logits, labs,
                    dice_ignore_index=args.ignore_index,
                    dice_weight=1.0, ce_weight_scale=1.0
                )

            # Backward pass (mixed precision if available)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss /= max(1, len(train_loader))

        # Validation pass
        model.eval()
        val_loss, dices = 0.0, []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                labs = batch['label'].to(device, non_blocking=True)
                logits = model(imgs)
                val_loss += ce_dice_loss(logits, labs, dice_ignore_index=args.ignore_index).item()
                d = per_class_dice_from_logits(
                    logits, labs,
                    num_classes=args.num_classes,
                    ignore_index=args.ignore_index
                )
                dices.append(torch.nanmean(d).item())

        val_loss /= max(1, len(val_loader))
        mean_val_dice = float(sum(dices) / max(1, len(dices))) if dices else 0.0

        scheduler.step(val_loss)
        early.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(mean_val_dice)
        save_curves(history, args.outdir)

        # Save checkpoint if model improved
        ckpt_path = Path(args.outdir) / 'best.pt'
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'args': vars(args)}, ckpt_path)
            logging.info(f"New best model saved (epoch {epoch}, val_loss={val_loss:.4f})")

        print(f'epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mean_dice={mean_val_dice:.4f}')
        logging.info(f'epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mean_dice={mean_val_dice:.4f}')

        if early.should_stop:
            print(f"early_stop_at_epoch={epoch}")
            logging.info(f"Early stopping at epoch {epoch}")
            break

    # Load best checkpoint for final testing
    best = torch.load(Path(args.outdir) / 'best.pt', map_location=device)
    model.load_state_dict(best['model'])
    model.eval()

    dices_per_class = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labs = batch['label'].to(device, non_blocking=True)
            logits = model(imgs)
            d = per_class_dice_from_logits(
                logits, labs,
                num_classes=args.num_classes,
                ignore_index=args.ignore_index
            )
            dices_per_class.append(d.unsqueeze(0))

    # Aggregate test Dice across all volumes
    if dices_per_class:
        d_all = torch.nanmean(torch.cat(dices_per_class, dim=0), dim=0)
        for c, val in enumerate(d_all.tolist()):
            print(f'class_{c}_dice={val:.4f}')
            logging.info(f'class_{c}_dice={val:.4f}')
        mean_dice = torch.nanmean(d_all).item()
        print(f'mean_dice_ex_bg={mean_dice:.4f}')
        logging.info(f'mean_dice_ex_bg={mean_dice:.4f}')


if __name__ == '__main__':
    main()
