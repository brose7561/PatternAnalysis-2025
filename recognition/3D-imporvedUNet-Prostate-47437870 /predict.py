import argparse
import os
import logging
from pathlib import Path
import io
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import imageio.v2 as imageio
from modules import UNet3DImproved
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
    p.add_argument('--checkpoint', type=str, default='runs_improved_unet3d/best.pt')
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--ignore_index', type=int, default=0)
    p.add_argument('--spatial_size', type=int, nargs=3, default=[128,128,64])
    p.add_argument('--outdir', type=str, default='pred_outputs')
    p.add_argument('--gif_fps', type=int, default=10)
    return p.parse_args()

def load_nifti(path):
    return nib.load(path)

def resolve_class_names(num_classes, ckpt_args):
    names = ckpt_args.get('class_names', None)
    if isinstance(names, (list, tuple)) and len(names) == num_classes:
        return [str(x) for x in names]
    return [f"class {i}" for i in range(num_classes)]

def load_model(checkpoint_path, device, num_classes):
    ckpt = torch.load(checkpoint_path, map_location=device)
    margs = ckpt.get('args', {})
    model = UNet3DImproved(
        in_channels=1,
        num_classes=num_classes,
        base_ch=margs.get('base_ch', 16),
        depth=margs.get('depth', 4),
        dropout_p=margs.get('dropout', 0.0)
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, margs

def normalize_image(img):
    return (img - img.mean()) / (img.std() + 1e-8)

def run_inference(model, img, device, spatial_size):
    img_rs = _resize_vol(img, tuple(spatial_size[::-1]), order=1)
    tens = torch.from_numpy(img_rs[None, None, ...]).float().to(device)
    with torch.no_grad():
        logits = model(tens)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    return img_rs, pred, logits

def build_discrete_cmap(num_classes):
    cmap = plt.cm.get_cmap('tab20', num_classes)
    bounds = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def legend_handles(present_classes, cmap, norm, class_names):
    handles = []
    for cid in sorted(present_classes):
        color = cmap(norm(cid))
        handles.append(mpatches.Patch(color=color, label=class_names[cid]))
    return handles

def frame_image(img_rs, pred_slice, label_slice, cmap, norm, class_names, present_classes):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=False)
    axes[0].imshow(img_rs, cmap='gray')
    axes[0].set_title('image')
    axes[0].axis('off')
    axes[1].imshow(pred_slice, cmap=cmap, norm=norm, interpolation='nearest')
    axes[1].set_title('pred')
    axes[1].axis('off')
    if label_slice is not None:
        axes[2].imshow(label_slice, cmap=cmap, norm=norm, interpolation='nearest')
        axes[2].set_title('label')
        axes[2].axis('off')
    else:
        axes[2].axis('off')
    fig.subplots_adjust(bottom=0.22)
    fig.legend(
        handles=legend_handles(present_classes, cmap, norm, class_names),
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(present_classes), 6),
        frameon=False,
        fontsize=8
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    arr = imageio.imread(buf)
    buf.close()
    return arr

def make_triptych_gif(img_vol, pred_vol, label_vol, num_classes, class_names, out_path, fps):
    cmap, norm = build_discrete_cmap(num_classes)
    present = set(np.unique(pred_vol))
    if label_vol is not None:
        present |= set(np.unique(label_vol))
    frames = []
    Z = img_vol.shape[2]
    for k in range(Z):
        g = img_vol[:, :, k]
        ps = pred_vol[:, :, k]
        ls = label_vol[:, :, k] if label_vol is not None else None
        frame = frame_image(g, ps, ls, cmap, norm, class_names, present)
        frames.append(frame)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)

def compute_dice(logits, gt_rs, device, num_classes, ignore_index):
    gt_t = torch.from_numpy(gt_rs[None, ...]).long().to(device)
    with torch.no_grad():
        d = per_class_dice_from_logits(logits, gt_t, num_classes=num_classes, ignore_index=ignore_index)
    for c, val in enumerate(d.tolist()):
        msg = f'class_{c}_dice={val:.4f}'
        print(msg)
        logging.info(msg)
    mean_dice = torch.nanmean(d).item()
    print(f'mean_dice_ex_bg={mean_dice:.4f}')
    logging.info(f"mean_dice_ex_bg={mean_dice:.4f}")

def save_nifti_like(src_img, data, out_path):
    nib.save(nib.Nifti1Image(data.astype(np.int16), src_img.affine, src_img.header), out_path)

def main():
    args = parse_args()
    device = to_device()
    os.makedirs(args.outdir, exist_ok=True)
    model, margs = load_model(args.checkpoint, device, args.num_classes)
    class_names = resolve_class_names(args.num_classes, margs)
    src_img = load_nifti(args.image_path)
    img = normalize_image(src_img.get_fdata().astype(np.float32))
    img_rs, pred, logits = run_inference(model, img, device, args.spatial_size)
    label_rs = None
    if args.label_path:
        gt = load_nifti(args.label_path).get_fdata().astype(np.int16)
        label_rs = _resize_vol(gt, tuple(args.spatial_size[::-1]), order=0)
    out_base = Path(args.outdir) / Path(args.image_path).stem
    gif_path = out_base.with_suffix('').as_posix() + "_triptych.gif"
    nii_path = out_base.with_suffix('').as_posix() + "_pred.nii.gz"
    make_triptych_gif(img_rs, pred, label_rs, args.num_classes, class_names, gif_path, args.gif_fps)
    save_nifti_like(src_img, pred, nii_path)
    print(f"saved {gif_path}")
    print(f"saved {nii_path}")
    logging.info(f"saved {gif_path}")
    logging.info(f"saved {nii_path}")
    if label_rs is not None:
        compute_dice(logits, label_rs, device, args.num_classes, args.ignore_index)

if __name__ == '__main__':
    main()
