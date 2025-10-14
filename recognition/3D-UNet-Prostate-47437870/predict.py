import argparse
import os
import logging
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from modules import UNet3D
from utils import to_device, per_class_dice_from_logits
from dataset import _resize_vol

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predict.log",
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_path', type=str, required=True)
    p.add_argument('--label_path', type=str, default=None)
    p.add_argument('--checkpoint', type=str, default='runs_3dunet/best.pt')
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--ignore_index', type=int, default=0)
    p.add_argument('--spatial_size', type=int, nargs=3, default=[128,128,64])
    p.add_argument('--outdir', type=str, default='pred_outputs')
    return p.parse_args()

def load_nifti(path):
    return nib.load(path).get_fdata()

def main():
    args = parse_args()
    device = to_device()
    os.makedirs(args.outdir, exist_ok=True)
    logging.info(f"Starting prediction for image {args.image_path} using checkpoint {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    margs = ckpt.get('args', {})
    in_ch = 1
    num_classes = args.num_classes
    base_ch = margs.get('base_ch', 16)
    depth = margs.get('depth', 4)
    dropout = margs.get('dropout', 0.0)

    model = UNet3D(in_channels=in_ch, num_classes=num_classes, base_ch=base_ch, depth=depth, dropout_p=dropout).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logging.info(f"Model loaded with base_ch={base_ch}, depth={depth}, dropout={dropout}")

    img = load_nifti(args.image_path).astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    img_rs = _resize_vol(img, tuple(args.spatial_size[::-1]), order=1)
    tens = torch.from_numpy(img_rs[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = model(tens)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    logging.info("Prediction completed successfully")

    mid = pred.shape[2] // 2
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img_rs[:,:,mid], cmap='gray')
    plt.title('image')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(pred[:,:,mid], interpolation='nearest')
    plt.title('pred')
    plt.axis('off')
    if args.label_path:
        gt = load_nifti(args.label_path).astype(np.int16)
        gt_rs = _resize_vol(gt, tuple(args.spatial_size[::-1]), order=0)
        plt.subplot(1,3,3)
        plt.imshow(gt_rs[:,:,mid], interpolation='nearest')
        plt.title('label')
        plt.axis('off')
    plt.tight_layout()
    out_png = Path(args.outdir) / 'prediction.png'
    plt.savefig(out_png)
    plt.close()
    print(f'saved {out_png}')
    logging.info(f"Saved prediction visualization to {out_png}")

    if args.label_path:
        gt_t = torch.from_numpy(gt_rs[None, ...]).long().to(device)
        with torch.no_grad():
            d = per_class_dice_from_logits(logits, gt_t, num_classes=num_classes, ignore_index=args.ignore_index)
        for c, val in enumerate(d.tolist()):
            msg = f'class_{c}_dice={val:.4f}'
            print(msg)
            logging.info(msg)
        mean_dice = torch.nanmean(d).item()
        print(f'mean_dice_ex_bg={mean_dice:.4f}')
        logging.info(f"mean_dice_ex_bg={mean_dice:.4f}")

if __name__ == '__main__':
    main()
