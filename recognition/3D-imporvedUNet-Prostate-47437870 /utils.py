# © Benjamin Rose, 2025
# please see attached licence - MIT
# Utility functions for device selection, Dice computation, combined CE+Dice loss,
# and saving training/validation metric plots.

import os
import torch
import matplotlib.pyplot as plt


def to_device():
    """Return the best available compute device (GPU if possible)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def per_class_dice_from_logits(logits, targets, num_classes, ignore_index=0, eps=1e-6):
    """Compute Dice score per class from raw model logits."""
    probs = torch.softmax(logits, dim=1)  # convert logits to probabilities
    preds = torch.argmax(probs, dim=1)    # predicted class labels
    dices = []

    for c in range(num_classes):
        if c == ignore_index:  # skip ignored class (usually background)
            dices.append(torch.tensor(float('nan'), device=logits.device))
            continue

        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()              # intersection area
        denom = p.sum() + t.sum()          # total area
        dice = (2 * inter + eps) / (denom + eps)
        dices.append(dice)

    return torch.stack(dices)


def ce_dice_loss(logits, targets, ce_weight=None, dice_ignore_index=0, eps=1e-6,
                 dice_weight=1.0, ce_weight_scale=1.0):
    """Combined cross-entropy and Dice loss for multi-class segmentation."""
    ce = torch.nn.functional.cross_entropy(logits, targets, weight=ce_weight)

    probs = torch.softmax(logits, dim=1)
    num_classes = logits.shape[1]

    # One-hot encode targets for Dice computation
    targets_oh = torch.nn.functional.one_hot(
        targets.clamp(min=0),
        num_classes=num_classes
    ).permute(0, 4, 1, 2, 3).float()

    if dice_ignore_index is not None:
        mask = (targets != dice_ignore_index).float()  # exclude ignored voxels
        probs = probs * mask.unsqueeze(1)
        targets_oh = targets_oh * mask.unsqueeze(1)

    inter = (probs * targets_oh).sum(dim=(0, 2, 3, 4))
    denom = probs.sum(dim=(0, 2, 3, 4)) + targets_oh.sum(dim=(0, 2, 3, 4))
    dice_per_class = (2 * inter + eps) / (denom + eps)
    dice = 1 - dice_per_class.mean()  # convert similarity to loss

    return ce_weight_scale * ce + dice_weight * dice


def save_curves(history, out_dir):
    """Save training and validation loss/dice curves to disk."""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss curves
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train')
    plt.plot(epochs, history['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    # Plot validation Dice curve if available
    if 'val_dice' in history:
        plt.figure()
        plt.plot(epochs, history['val_dice'], label='mean_dice')
        plt.xlabel('epoch')
        plt.ylabel('dice')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'dice.png'))
        plt.close()
