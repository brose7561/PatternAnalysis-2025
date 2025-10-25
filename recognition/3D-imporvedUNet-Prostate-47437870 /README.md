

# Improved UNet3D for Prostate MRI Segmentation

![](pictures/full_data_10_epochs.gif)


## Overview

This repository trains and evaluates an **Improved UNet3D** for 3D prostate MRI semantic segmentation on downsampled NIfTI volumes. The goal is to achieve **≥ 0.70 Dice similarity coefficient (DSC)** for *all* foreground classes. The model extends a pytorch 3D U-Net with **Dynamic ReLU (Dy-ReLU)** activations and **CBAM** attention.

> Why this repo exists: To increment the reasearch. Providing a stable and extensibe solusion for 3d segmentation. 

---

## How It Works (Short)

* **Backbone:** 3D U-Net encoder–decoder with skip connections. 


* **Blocks:** Residual blocks with `BatchNorm3d → DyReLU3d → Conv3d` ×2 (+ identity).
* **Attention:** **CBAM3d** (channel then spatial attention) applied to **skip features** before decoding.
* **Head:** 1×1×1 conv to class logits.
* **Loss:** Combined **Cross-Entropy + soft Dice**.
* **Metrics:** Per-class Dice and mean Dice (excluding background).
* **I/O:** NIfTI volumes (`.nii/.nii.gz`) with z-score normalization; labels are nearest-neighbor resized.

---

## Model Architecture (Key Ideas)

### 1) Dy-ReLU for 3D

We use a per-channel, input-conditioned activation (**K=2** piecewise linear units) to increase representational capacity without large compute overhead. It generates `(a_k, b_k)` from the global pooled feature and computes `max_k(a_k·x + b_k)`.

Reference: **Dynamic ReLU** (PMCID: **PMC8793173**) — discusses stability issues from inconsistent mini-batch statistics and motivation for trainable, input-adaptive activations. Less noise more training needed. 

### 2) CBAM3d on Skips

**Channel + Spatial attention** (3D) enhances informative skip features **before** concatenation with the upsampled decoder signal, improving boundary detail and suppressing noise.

### 3) Residual Stages

Each stage uses **residual connections** to ease optimization and help gradients flow in deeper 3D stacks.

> **Small-batch stability note.** 3D segmentation often runs at **batch size = 1** due to memory limits. This can make **BatchNorm** statistics noisy and cause “salt-and-pepper” pixel scatter in early training (as observed locally). Two mitigations used here:
>
> * Adopt **Dy-ReLU** (trainable, input-conditioned) to soften dependence on batch statistics.
> * (Optional variant) **Filter Response Normalization (FRN)** + **TLU** can fully remove batch dependence. We kept `BatchNorm3d` in the final model but almost stuck with the above when test training locally with small batches. 
---

## Repository Layout

```
modules.py     # DyReLU3d, CBAM3d, residual blocks, UNet3DImproved
dataset.py     # NIfTI dataset, pairing, resizing, augmentation, DataLoaders
train.py       # training loop, early stopping, LR scheduler, curve saving
predict.py     # single-volume inference, GIF triptych, NIfTI export, Dice
utils.py       # device utils, dice/loss, curve plotting
requirements.txt
pictures/      # figures and GIFs used in this README
```

---

## Figures & Training Snapshots

**Local smoke test train (note tiny test sizes)**
![](pictures/firstLocaltraining.png)

7 scans for 5 epochs. 

**Local test -— note scatter:**
![](pictures/first_Local.gif)

The scatter was not present when using the orrigional unet in the same conditions. 

**Local test with FRN TLU (batch-independent norm) :**
![](pictures/local_RFN_remove_batch.gif)

Worried about the scatter i branched out and used FRN and TLU after finding similar projects reporting scattered results in small batch sizes. The scatter reduced a lot, however so did the mean dice scores.  

**Full-data cluster run (10 epochs) — qualitative convergence:**
![](pictures/full_data_10_epochs.gif)

In the end i returned to the orrigional batchnorm, combo and trained against the full dataset. 

# Training Results

> **Why 10 epochs?** The task requires **DSC ≥ 0.70** across foreground classes. Training substantially longer is a great project extension, but would waste resourses in the context of this projects requirments.  

**Dice values:**
![](pictures/final_training_dice.png)

**Dice Curve:**
![](pictures/dice.png)

**Loss Curve:**
![](pictures/loss.png)

---

## Usage

### 1) Install

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate #for mac
pip install -r requirements.txt
```


### 2) Data

Organize the downsampled **HipMRI / Prostate** data as:

```
/path/to/images/*.nii[.gz]
/path/to/labels/*.nii[.gz]
# Filenames must match by stem (e.g., case_00001.nii in both).
```

### 3) Train

```bash
python train.py \
  --image_dir /path/to/images \
  --label_dir /path/to/labels \
  --spatial_size <int int int> \
  --epochs <int> \
  --batch_size <int> \
  --lr 4e-4 \
  --num_workers <int> \
  --num_classes <int> \
  --ignore_index <int> \
  --outdir runs_improved_unet3d \
  --augment            # optional: enable simple flips/rot90
```
#### running UQ Rangpur is default just run:
```bash
python train.py \
  --num_workers 1 \
  --epochs 10
```

#### During training, curves are saved to:

```
runs_improved_unet3d/loss.png
runs_improved_unet3d/dice.png
runs_improved_unet3d/best.pt
```

### 4) Inference (and Triptych GIF)

```bash
python predict.py \
  --image_path /path/to/test_volume.nii.gz \
  --label_path /path/to/test_label.nii.gz \   # optional, enables per-class Dice
  --checkpoint runs_improved_unet3d/best.pt \
  --num_classes <int> \
  --ignore_index <int> \
  --spatial_size <int> <int> <int> \
  --outdir pred_outputs \
  --gif_fps <int>
```

#### Default example:

```bash
python predict.py \
--image_path ./testdata/semantic_MRs/B006_Week0_LFOV.nii.gz \
--label_path ./testdata/semantic_labels_only/B006_Week0_SEMANTIC.nii.gz \
--checkpoint ./runs_localtest/best.pt

```

#### This writes:

```
pred_outputs/test_volume_triptych.gif   # image | prediction | (optional) label
pred_outputs/test_volume_pred.nii.gz    # predicted labelmap
```



## Pre-processing & Augmentation

* **Z-score normalization** per-volume:

[
x' = \frac{x - \mu}{\sigma + 1e!-!8}
]

* **Resizing** to `spatial_size` (default **128×128×64**):

  * Images: **trilinear** interpolation
  * Labels: **nearest-neighbor** (to preserve integers)
* **Augmentations** (train only, lightweight & geometry-safe):

  * Random flips on each axis (p = 0.5)
  * Random 90° rotations on random axis pair (p ≈ 0.5)

> These augmentations are chosen to be label-preserving and memory-light, in line with the task’s focus on a strong, simple baseline.

---

## Data Splits & Reproducibility

* **Default splits:** `train/val/test = 80% / 10% / 10%` (rounded)
* **Deterministic split** via `torch.Generator().manual_seed(args.seed)`
* **Justification:**

  * 10% validation is sufficient to guide early stopping and LR scheduling without starving training.
  * 10% hold-out test is adequate to verify the **≥ 0.70 DSC** requirement while preserving most data for fitting.
  * Fixed seed ensures repeatable pairing/shuffling and consistent comparisons across runs.

---

## Training Choices & Rationale

* **Epochs:** **10** by default. The assignment targets **≥ 0.70 DSC**; this setup reaches the bar quickly. Extending training would consume more compute for marginal benefit **relative to task requirements**.
* **Loss:** **CE + soft Dice** balances region coverage and boundary overlap.
* **Optimizer:** `Adam(lr=4e-4, weight_decay=1e-4)`; `ReduceLROnPlateau` for stability.
* **Early stopping:** Patience (default **25**) safeguards against overfitting and wasted epochs. In the end I only trained 10 epochs to easily pass DSC 0.7
* **Small-batch normalization:** If you see early “static”/scatter in predictions (common at batch=1), consider:

  1. keep Dy-ReLU (already helps), and/or
  2. swap `BatchNorm3d` for **FRN+TLU** in the residual blocks for fully batch-independent normalization in extremely constrained memory regimes.

---

## Results (Qualitative Summary)

* **Local quick runs** (few subjects / few epochs): visible “pixel scatter” early on due to unstable statistics and under-training (see *Local* figures).
* **Full data, ~10 epochs:** clean structures, reduced speckle, and test-set Dice meeting/exceeding **0.70** across foreground classes (see GIF and curves).
* **Triptych GIFs** produced by `predict.py` show **image | prediction | label** for fast sanity checks.

---

## References

* **Dynamic ReLU** (context & stability under batch variations): *A Comprehensive Study of Swish and a Proposal of New Activation Function (DyReLU)* — PMCID: **PMC8793173**.
* **U-Net**: Ronneberger et al., 2015 (3D adaptation used here).
* **CBAM**: Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018.
* **(Optional) FRN**: Singh & Krishnan, *Filter Response Normalization Layer*, CVPR 2020.

---

## Citation

If you build on this codebase in academic work, please cite the above methods and include a link to this repository in your acknowledgements.

---

## License

MIT 

