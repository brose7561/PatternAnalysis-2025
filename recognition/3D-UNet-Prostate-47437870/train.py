import os
import argparse
import math
import logging
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from modules import UNet3D
from dataset import make_loaders
from utils import to_device, per_class_dice_from_logits, ce_dice_loss, save_curves

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', type=str, default='/home/groups/comp3710/HipMRI_Study_open/semantic_MRs')
    p.add_argument('--label_dir', type=str, default='/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only')
    p.add_argument('--spatial_size', type=int, nargs=3, default=[128,128,64])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base_ch', type=int, default=16)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--ignore_index', type=int, default=0)
    p.add_argument('--outdir', type=str, default='runs_3dunet')
    p.add_argument('--augment', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    device = to_device()
    os.makedirs(args.outdir, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(
        args.image_dir, args.label_dir,
        spatial_size=tuple(args.spatial_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1, test_split=0.1,
        seed=args.seed,
        augment_train=args.augment
    )
    model = UNet3D(in_channels=1, num_classes=args.num_classes, base_ch=args.base_ch, depth=args.depth, dropout_p=args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_val = math.inf
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}

    logging.info(f"Starting training for {args.epochs} epochs on device {device}.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')
        for batch in pbar:
            imgs = batch['image'].to(device, non_blocking=True)
            labs = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = ce_dice_loss(logits, labs, dice_ignore_index=args.ignore_index, dice_weight=1.0, ce_weight_scale=1.0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device, non_blocking=True)
                labs = batch['label'].to(device, non_blocking=True)
                logits = model(imgs)
                val_loss += ce_dice_loss(logits, labs, dice_ignore_index=args.ignore_index).item()
                d = per_class_dice_from_logits(logits, labs, num_classes=args.num_classes, ignore_index=args.ignore_index)
                dices.append(torch.nanmean(d).item())
        val_loss /= max(1, len(val_loader))
        mean_val_dice = float(sum(dices) / max(1, len(dices))) if dices else 0.0
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(mean_val_dice)

        ckpt_path = Path(args.outdir) / 'best.pt'
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(),
                        'args': vars(args)}, ckpt_path)
            logging.info(f"New best model saved at epoch {epoch} with val_loss={val_loss:.4f}")

        log_msg = f'epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mean_dice={mean_val_dice:.4f}'
        print(log_msg)
        logging.info(log_msg)

        save_curves(history, args.outdir)

    best = torch.load(Path(args.outdir) / 'best.pt', map_location=device)
    model.load_state_dict(best['model'])
    model.eval()
    dices_per_class = []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device, non_blocking=True)
            labs = batch['label'].to(device, non_blocking=True)
            logits = model(imgs)
            d = per_class_dice_from_logits(logits, labs, num_classes=args.num_classes, ignore_index=args.ignore_index)
            dices_per_class.append(d.unsqueeze(0))
    if dices_per_class:
        d_all = torch.nanmean(torch.cat(dices_per_class, dim=0), dim=0)
        for c, val in enumerate(d_all.tolist()):
            logging.info(f'class_{c}_dice={val:.4f}')
            print(f'class_{c}_dice={val:.4f}')
        mean_dice = torch.nanmean(d_all).item()
        logging.info(f'mean_dice_ex_bg={mean_dice:.4f}')
        print(f'mean_dice_ex_bg={mean_dice:.4f}')

if __name__ == '__main__':
    main()
