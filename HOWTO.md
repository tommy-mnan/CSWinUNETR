# How to Run CSWinUNETR

This guide covers setting up, preparing data for, training, and evaluating
CSWinUNETR on the FFHQ-Wrinkle dataset. It also explains what each part of the
code does, so you can tell what is happening at every step and where to look
when something fails.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [What Is (and Is Not) in the Repository](#2-what-is-and-is-not-in-the-repository)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Preparing the Dataset](#5-preparing-the-dataset)
6. [Checking the Dataset Before Training](#6-checking-the-dataset-before-training)
7. [Training](#7-training)
8. [Inference and Evaluation](#8-inference-and-evaluation)
9. [End-to-End Project Flow](#9-end-to-end-project-flow)
10. [Code Walkthrough](#10-code-walkthrough)
11. [Using the Model in Your Own Code](#11-using-the-model-in-your-own-code)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What This Project Does

CSWinUNETR is a segmentation network for **thin, curvy structures** such as
retinal vessels, brain blood vessels and facial wrinkles. Structures like these
are hard to segment for three reasons:

- They are **thin**, often only 1–3 pixels wide, so downsampling loses them easily.
- They are **long and winding**, so the model needs context that spans the whole image.
- They are **rare**: most pixels are background, which leaves the classes heavily imbalanced.

The model attacks these problems with three ideas:

| Component | Problem it solves |
|---|---|
| **Cross-shaped stripe attention** with cyclic shifts | Each pixel attends along long horizontal and vertical stripes, which captures long-range context along the main axes where thin structures run |
| **Detail-enhanced multi-scale attention** | Keys and values come from several resolutions, so fine detail survives next to global context |
| **Sparse-control dynamic snake convolution (SDSConv)** | A convolution kernel that bends to follow curved structures. It predicts a few control points and interpolates a smooth, dense kernel path from them |

The repository includes an example pipeline for **2D binary segmentation of
facial wrinkles** on FFHQ-Wrinkle. The model itself also supports 3D input and
multiclass segmentation.

---

## 2. What Is (and Is Not) in the Repository

| Item | Included? | Notes |
|---|---|---|
| Model code (`cswinunetr/models/`) | ✅ Yes | 2D and 3D, binary and multiclass |
| Training and inference scripts | ✅ Yes | `cswinunetr/train.py`, `cswinunetr/inference.py` |
| Split files (train/val/test IDs) | ✅ Yes | `cswinunetr/datasets/splits/ffhq/`: 800 / 100 / 100 IDs |
| **Dataset images and masks** | ❌ **No** | Download them yourself (see [Section 5](#5-preparing-the-dataset)) |
| **Pretrained checkpoint** | ❌ **No** | Train your own model (see [Section 7](#7-training)) |

`data/`, `runs/`, `outputs/` and every `*.pt` file are listed in `.gitignore`,
so datasets and checkpoints you create are never committed by accident.

---

## 3. Requirements

### Software

- **Python 3.10 or 3.11.** The project requires `>=3.10`. On macOS the system
  `/usr/bin/python3` is 3.9, which is too old. Use Homebrew's
  `/opt/homebrew/bin/python3.11` or another 3.10+ interpreter.
- **PyTorch 2.5.x.** `pyproject.toml` pins `torch>=2.5,<2.6`.
- The remaining dependencies install automatically (see below).

### Hardware

| Task | Recommended hardware |
|---|---|
| Full training (1024×1024, batch 6) | NVIDIA GPU with a lot of memory (roughly 24 GB or more). Lower `--batch-size` and/or add `--use-checkpoint` on smaller GPUs |
| Inference | Any CUDA GPU. CPU works but is slow at 1024×1024 |
| Quick code test | CPU is fine with small inputs |

> **Apple Silicon (Mac) users:** the code targets `cuda` and `cpu`. You can try
> `--device mps`, but this path has not been tested, and some operations may be
> slow or unsupported. For real training, use a CUDA machine such as a lab
> server or a cloud GPU.

---

## 4. Installation

```bash
# 1. Go to the project folder
cd CSWinUNETR

# 2. Create a virtual environment with Python 3.10+
python3.11 -m venv .venv

# 3. Activate it (run this in every new terminal)
source .venv/bin/activate

# 4. Install the package and its dependencies
python -m pip install --upgrade pip
python -m pip install -e '.[train,eval]'
```

### What the install options mean

| Command | Installs | Use when |
|---|---|---|
| `pip install -e .` | `torch`, `monai`, `numpy`, `Pillow` | You only run inference |
| `pip install -e '.[train]'` | + `albumentations`, `opencv-python-headless`, `schedulefree` | You train models |
| `pip install -e '.[eval]'` | + `scipy`, `scikit-image` | You want clDice, Betti error and HD95 |

`-e` stands for "editable": edits you make to the code take effect without reinstalling.

> **Important:** `albumentations` is pinned to `1.3.1` on purpose. The
> augmentation code uses arguments such as `alpha_affine` that newer versions
> removed. Do not upgrade it.

> **CUDA builds:** if `pip` installs a CPU-only PyTorch on your Linux GPU
> machine, first install the matching CUDA build from
> <https://pytorch.org/get-started/previous-versions/> (for example
> `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`),
> then run the install command above.

### Check that the install works

This runs the model on random noise. No dataset is needed:

```bash
python -c "
import torch
from cswinunetr import CSWinUNETR
model = CSWinUNETR(in_channels=4, out_channels=2, spatial_dims=2).eval()
with torch.inference_mode():
    out = model(torch.randn(1, 4, 64, 64))
print('Output shape:', out.shape)
print('Parameters:', sum(p.numel() for p in model.parameters()))
"
```

Expected output: `Output shape: torch.Size([1, 2, 64, 64])`. The two output
channels are the background and wrinkle logits.

---

## 5. Preparing the Dataset

The dataset comes from a separate repository:
**<https://github.com/labhai/ffhq-wrinkle-dataset>**

Follow its instructions to:

1. Download the **manual wrinkle masks** (1,000 hand-labelled images).
2. Download the **weak wrinkle masks** (texture maps).
3. Download the original **FFHQ face images** (from NVIDIA's FFHQ release).
4. Run that repository's preprocessing to create the **masked face images**.

Check the dataset's license before you use it.

### Required folder layout

```text
data/
├── masked_face_images/       # INPUT: RGB face images
│   ├── 00001.png
│   └── ...
├── weak_wrinkle_masks/       # INPUT: grayscale texture maps (4th input channel)
│   └── 00000/                #   subfolders are allowed and searched recursively
│       ├── 00001.png
│       └── ...
└── manual_wrinkle_masks/     # LABEL: binary wrinkle masks (ground truth)
    ├── 00001.png
    └── ...
```

You can put `data/` anywhere and pass its path with `--data-root`.

### What each folder is for

| Folder | Role | Format |
|---|---|---|
| `masked_face_images` | Model input, channels 1–3 | RGB PNG, 1024×1024 |
| `weak_wrinkle_masks` | Model input, channel 4: a texture hint that guides the model | Grayscale PNG, 1024×1024 |
| `manual_wrinkle_masks` | Training target and evaluation ground truth | PNG, 1024×1024; any pixel `> 0` counts as wrinkle |

Note that the "weak wrinkle masks" are **not** used as labels. They are fed to
the model as an **extra input channel**. The model always takes **4 channels**
(R, G, B, texture), during both training and inference.

### How files are matched

Files are matched across folders by **filename without the extension**. For
example, `masked_face_images/00264.png`, `weak_wrinkle_masks/00000/00264.png`
and `manual_wrinkle_masks/00264.png` all belong to sample `00264`.

The split files (`train.txt`, `val.txt`, `test.txt`) list these IDs, one per
line. The test split matches the official `test_file_lists.txt` from the
dataset repository, so your results are comparable with the paper.

---

## 6. Checking the Dataset Before Training

The loader (`cswinunetr/datasets/ffhq_wrinkle.py`) checks the data strictly and
stops with an error if anything is wrong. Make sure that:

- [ ] All three folders exist inside `--data-root`, with exactly these names.
- [ ] Every file is a **`.png`**. Other formats are ignored.
- [ ] Each ID in `train.txt`, `val.txt` and `test.txt` has a file in **all three** folders.
- [ ] No ID appears twice within one folder, including subfolders.
- [ ] Every image, texture and mask is exactly **1024×1024**, or whatever `--image-size` you use.
- [ ] Each image, its texture map and its mask have the **same size**.
- [ ] Masks are black (0) for background and non-zero for wrinkle.

### Quick check script

Run this from the repository root after activating the venv. It catches most
problems in a few seconds:

```bash
python - <<'EOF'
from pathlib import Path
from PIL import Image
from cswinunetr.datasets.ffhq_wrinkle import index_pngs, read_ids, validate_splits, SPLIT_DIR

DATA = Path("/path/to/data")   # <-- change this

print("Split sizes:", validate_splits())          # also checks that splits do not overlap
folders = {
    "images":   index_pngs(DATA / "masked_face_images"),
    "textures": index_pngs(DATA / "weak_wrinkle_masks"),
    "masks":    index_pngs(DATA / "manual_wrinkle_masks"),
}
for name, files in folders.items():
    print(f"{name:9s}: {len(files)} PNG files")

for split in ("train", "val", "test"):
    ids = read_ids(SPLIT_DIR / f"{split}.txt")
    for name, files in folders.items():
        missing = [i for i in ids if i not in files]
        if missing:
            print(f"[{split}] missing {len(missing)} {name}, e.g. {missing[:5]}")

sample = read_ids(SPLIT_DIR / "train.txt")[0]
for name, files in folders.items():
    with Image.open(files[sample]) as im:
        print(f"{sample} {name}: size={im.size} mode={im.mode}")
EOF
```

If no `missing` lines are printed and every size is `(1024, 1024)`, the data is
ready.

---

## 7. Training

### Basic command

Run from the repository root:

```bash
python cswinunetr/train.py \
  --data-root /path/to/data \
  --output-dir runs/ffhq \
  --device cuda
```

### What happens during training

1. **Seeding:** Python, NumPy and PyTorch are seeded (`--seed 42`) so runs are reproducible.
2. **Split validation:** the script checks that train, val and test do not overlap.
3. **Data loading:** 800 training images (augmented) and 100 validation images (not augmented).
4. **Model creation:** CSWinUNETR with 4 input channels and 2 output classes.
5. **Training loop:** for each epoch:
   - Every batch passes through the model, the **Dice + Cross-Entropy** loss is computed, and the **schedule-free AdamW** optimizer updates the weights.
   - After each step, an **EMA (exponential moving average)** copy of the weights is updated with decay 0.9999. The EMA weights are smoother and usually generalize better.
6. **Validation:** every 5 epochs, the **EMA model** is scored on the validation set with Dice.
7. **Checkpointing:**
   - When validation Dice improves, the EMA weights are saved to `best.pt`.
   - After every epoch, the full state is saved to `last.pt` for resuming.
8. **Early stopping:** training stops if validation Dice fails to improve for 30 validation checks in a row (`--patience`), which is 150 epochs at the default interval, or when it reaches 1000 epochs.

### Output files (`runs/ffhq/`)

| File | Contents | Used for |
|---|---|---|
| `best.pt` | EMA weights with the best validation Dice, plus the model config | **Inference** |
| `last.pt` | Model, EMA, optimizer, RNG state, epoch counters | **Resuming** training |
| `metrics.jsonl` | One JSON line per epoch: loss, `val_dice`, timing, GPU memory | Monitoring and plotting |
| `settings.json` | Every command-line argument used | Reproducibility |
| `environment.json` | PyTorch/CUDA version, GPU name, parameter count | Reproducibility |
| `splits/` | Copies of the split files used | Checking that resumed runs use the same splits |

Watch progress live:

```bash
tail -f runs/ffhq/metrics.jsonl
```

### Main training arguments

| Argument | Default | Meaning |
|---|---|---|
| `--data-root` | *(required)* | Folder that contains the three data folders |
| `--output-dir` | `runs/ffhq` | Where results are saved. **Must be empty** unless you pass `--resume` |
| `--device` | `cuda` if available, else `cpu` | `cuda`, `cuda:1`, `cpu`, etc. |
| `--epochs` | `1000` | Maximum number of epochs |
| `--batch-size` | `6` | Images per step. Lower it if you run out of GPU memory |
| `--lr` | `1e-3` | Learning rate |
| `--image-size` | `1024` | Must match the real image size; multiple of 32, at least 64 |
| `--workers` | `4` | Data-loading processes. Use `0` to debug |
| `--eval-interval` | `5` | Validate every N epochs |
| `--patience` | `30` | Validation checks without improvement before stopping |
| `--use-checkpoint` | off | Gradient checkpointing: recomputes encoder activations, so it uses less GPU memory but runs slower |
| `--resume` | off | Continue from `output-dir/last.pt` |
| `--no-progress` | off | Hide the progress bars and print one JSON line per epoch instead (useful for log files) |
| `--split-dir` | built-in FFHQ splits | Folder with your own `train.txt` / `val.txt` / `test.txt` |
| `--seed` | `42` | Random seed |

### Model architecture arguments (advanced)

| Argument | Default | Meaning |
|---|---|---|
| `--feature-size` | `48` | Base channel width. It doubles at each stage: 48 → 96 → 192 → 384 → 768 |
| `--depths` | `2 2 2 2` | Number of transformer blocks per encoder stage |
| `--num-heads` | auto (`2 4 8 16` in 2D) | Attention heads per stage |
| `--stripe-widths` | `1 2 7 7` | Stripe width of the cross-shaped attention per stage |
| `--pool-ratios` | `8 4 2 1` | Downsampling ratios for the multi-scale keys/values |
| `--strip-kernel-sizes` | `9 7 5 3` | Axial depthwise kernel sizes (odd numbers) |
| `--sds-stages` | `1 0 0 0` | Which stages use snake convolution (1 = on). Default: the first stage only |
| `--sds-kernel-sizes` | `11 9 7 5` | Snake kernel lengths (odd, at least 5) |
| `--mlp-ratio` | `4.0` | Transformer MLP width multiplier |
| `--drop-path-rate` | `0.2` | Maximum stochastic-depth rate |

Run `python cswinunetr/train.py --help` to see the full list.

### Out of GPU memory?

Try these in order:

```bash
--batch-size 4
--batch-size 2 --use-checkpoint
--batch-size 1 --use-checkpoint
```

### Resuming an interrupted run

Rerun the **exact same command** and add `--resume`:

```bash
python cswinunetr/train.py \
  --data-root /path/to/data \
  --output-dir runs/ffhq \
  --device cuda \
  --resume
```

The script refuses to resume if important settings changed (batch size, image
size, learning rate, model architecture, data root or split files). This keeps
a resumed run consistent with the original.

### Quick smoke test (CPU, tiny run)

To check the pipeline end to end without a big GPU, make downscaled copies of
the data (for example 256×256, with nearest-neighbour resizing for masks) and run:

```bash
python cswinunetr/train.py \
  --data-root /path/to/data_256 \
  --output-dir runs/smoke \
  --device cpu \
  --image-size 256 --batch-size 1 --workers 0 \
  --epochs 2 --eval-interval 1
```

Scores from this run are meaningless. It only confirms that everything runs.

---

## 8. Inference and Evaluation

### Evaluate on the test set

```bash
python cswinunetr/inference.py \
  --checkpoint runs/ffhq/best.pt \
  --image-dir   /path/to/data/masked_face_images \
  --texture-dir /path/to/data/weak_wrinkle_masks \
  --mask-dir    /path/to/data/manual_wrinkle_masks \
  --ids-file    cswinunetr/datasets/splits/ffhq/test.txt \
  --output-dir  outputs/ffhq-test \
  --device cuda
```

### Add thin-structure metrics

Install the eval extras (`pip install -e '.[eval]'`), then add:

```bash
  --all-metrics
```

### Predict on unlabeled images

Leave out `--mask-dir`, and `--ids-file` too if you want every image in the folder:

```bash
python cswinunetr/inference.py \
  --checkpoint runs/ffhq/best.pt \
  --image-dir   /path/to/my_faces \
  --texture-dir /path/to/my_texture_maps \
  --output-dir  outputs/my-predictions
```

Even unlabeled images need a matching **texture map** with the same filename.

### Inference arguments

| Argument | Required | Meaning |
|---|---|---|
| `--checkpoint` | ✅ | Path to `best.pt`. The model config is read from the file |
| `--image-dir` | ✅ | RGB images |
| `--texture-dir` | ✅ | Texture maps |
| `--mask-dir` | optional | Ground-truth masks. Enables scoring |
| `--ids-file` | optional | Only process these IDs (for example `test.txt`) |
| `--output-dir` | optional | Default `outputs/ffhq`. **Must be empty** |
| `--device` | optional | `cuda` / `cpu` |
| `--all-metrics` | optional | Adds clDice, Betti error and HD95. Requires `--mask-dir` |

### Output files

| File | Contents |
|---|---|
| `<id>.png` | Predicted mask: pixel value **0** = background, **1** = wrinkle |
| `predictions.json` | Scores per image, mean scores, speed and memory stats |

The prediction PNGs look **completely black** in a normal image viewer because
1 is almost black. To view one:

```bash
python -c "
from PIL import Image; import numpy as np
m = np.array(Image.open('outputs/ffhq-test/00001.png'))
Image.fromarray((m * 255).astype('uint8')).save('preview.png')
"
```

### Understanding the metrics

| Metric | Better | What it measures |
|---|---|---|
| **Dice** | Higher (0–1) | Overlap between prediction and ground truth. The main metric |
| **clDice** | Higher (0–1) | Overlap of the *skeletons* (centerlines). Rewards keeping thin lines connected |
| **Betti-0 error** | Lower | Difference in the number of separate pieces. Measures fragmentation |
| **Betti-1 error** | Lower | Difference in the number of holes or loops |
| **Betti error** | Lower | Betti-0 + Betti-1 |
| **HD95** | Lower (pixels) | 95th-percentile boundary distance. Measures how far predictions stray |

---

## 9. End-to-End Project Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 1. DATA (you download)                                              │
│    masked_face_images/  +  weak_wrinkle_masks/  +  manual_wrinkle_  │
│         (RGB)                 (texture)              masks/ (label) │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  datasets/ffhq_wrinkle.py
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. LOADING & PREPROCESSING                                          │
│    • Match files by ID from train/val/test.txt                      │
│    • Train only: augmentation (flip, random crop, affine,           │
│      elastic/grid/optical distortion, brightness/contrast/hue)      │
│    • Stack RGB + texture → 4×1024×1024 tensor, scale to [-1, 1]     │
│    • Mask → 0/1 label map                                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. MODEL: CSWinUNETR (models/)                                      │
│                                                                     │
│   Input (4ch) ─► Conv embedding (7×7, stride 2)                     │
│                     │                                               │
│   Encoder ── Stage 1: SDSConv + CSWin blocks (stripe width 1)       │
│              Stage 2: CSWin blocks           (stripe width 2)       │
│              Stage 3: CSWin blocks           (stripe width 7)       │
│              Stage 4: global attention                              │
│                     │   each stage halves resolution, doubles ch.   │
│   Bottleneck ──► residual conv block                                │
│                     │                                               │
│   Decoder ── 5 upsampling blocks, each joined with a skip          │
│              connection from the matching encoder level             │
│                     ▼                                               │
│   Output: 2 × 1024 × 1024 logits (background, wrinkle)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TRAINING (engine/trainer.py)                                     │
│    loss = 0.5 × Dice loss + 0.5 × Cross-Entropy  (utils/losses.py)  │
│    optimizer = Schedule-Free AdamW (lr 1e-3, weight decay 0.05)     │
│    EMA weights (decay 0.9999) updated every step                    │
│    every 5 epochs: validate EMA → if best Dice → save best.pt       │
│    every epoch: save last.pt; early-stop after 30 stale checks      │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. INFERENCE (engine/inference.py)                                  │
│    load best.pt → predict each image → argmax → save <id>.png       │
│    if masks are given: Dice (+ clDice, Betti, HD95) → predictions.json │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Code Walkthrough

```text
cswinunetr/
├── __init__.py              # Exposes CSWinUNETR for `from cswinunetr import CSWinUNETR`
├── train.py                 # Thin entry point → engine/trainer.py:main()
├── inference.py             # Thin entry point → engine/inference.py:main()
├── models/
│   ├── cswinunetr.py        # Full network: embedding, encoder stages, decoder
│   ├── attention.py         # Cross-shaped stripe attention, cyclic shift, multi-scale K/V
│   └── sdsconv.py           # Sparse-control dynamic snake convolution
├── datasets/
│   ├── ffhq_wrinkle.py      # Dataset class, file indexing, augmentation, tensor conversion
│   └── splits/ffhq/         # train.txt (800), val.txt (100), test.txt (100)
├── engine/
│   ├── trainer.py           # Training loop, EMA, validation, checkpoints, resume
│   └── inference.py         # Prediction, PNG saving, metric computation
└── utils/
    ├── losses.py            # dice_ce_loss: Dice + Cross-Entropy
    └── metrics.py           # foreground_dice, thin_structure_metrics (clDice, Betti, HD95)
```

### `models/cswinunetr.py`: the network

- **`ConvEmbedding`** turns the input image into features at half resolution with a 7×7 convolution.
- **`EncoderStage`** (×4): each stage runs a residual conv block, optionally adds an **SDSConv**
  branch, then runs **CSWin attention blocks** (every second block is cyclically shifted), then halves the resolution and doubles the channels. Stage 4 uses
  global attention instead of stripes.
- **Skip connections**: the raw input and the first three encoder outputs pass through
  residual conv blocks and are handed to the decoder.
- **Decoder**: MONAI `UnetrUpBlock`s upsample step by step back to full resolution, merging each skip.
- **`UnetOutBlock`**: a final 1×1 conv that produces one logit map per class.

### `models/attention.py`: cross-shaped stripe attention

- The attention heads split into **two groups**: one attends along **horizontal stripes**
  and the other along **vertical stripes** (three groups in 3D). Together they form a
  "cross" around every pixel, so information travels along whole rows and columns.
- **Cyclic shift** shifts the feature map between blocks, so neighbouring stripes exchange information.
- **`MultiScaleKV`** pools keys and values to coarser resolutions (`pool_ratios`) and refines
  them with axial strip convolutions. Fine and coarse context are combined this way.

### `models/sdsconv.py`: snake convolution

- A normal convolution samples on a rigid grid. SDSConv first predicts a **few control
  points**, then **interpolates a smooth curved path** of sampling locations between them,
  so the kernel can follow a wrinkle or vessel as it bends.
- By default it is only used in **stage 1**, where resolution is highest and thin details matter most.

### `datasets/ffhq_wrinkle.py`: data

- `index_pngs()` recursively finds all PNGs and indexes them by filename stem.
- `FFHQWrinkle.__getitem__()` loads RGB, texture and mask, applies augmentation (train only),
  and returns `{"id", "image": 4×H×W float tensor in [-1, 1], "mask": H×W long tensor}`.
- Spatial augmentations transform the image, mask and texture together so they stay aligned.
  Color augmentations change the RGB image only.

### `engine/trainer.py`: training

- Parses arguments, validates them, builds the data loaders, model, optimizer and EMA.
- Saves `settings.json` and `environment.json` for reproducibility.
- Main loop: forward → loss → backward → optimizer step → EMA update, then validation and checkpoints.
- On resume, it restores the model, EMA, optimizer, RNG states and counters, and trims
  `metrics.jsonl` rows written after the last saved epoch.

### `utils/losses.py`: loss

`dice_ce_loss = 0.5 × (1 − Dice) + 0.5 × CrossEntropy`

- **Dice** addresses class imbalance: it measures overlap, not per-pixel accuracy, so the
  huge background does not dominate.
- **Cross-entropy** provides stable per-pixel gradients.

### `utils/metrics.py`: evaluation

- `foreground_dice()` computes Dice per image. If both prediction and ground truth are empty, the score is 1.
- `thin_structure_metrics()` computes clDice, Betti errors and HD95 with SciPy and scikit-image.

---

## 11. Using the Model in Your Own Code

```python
import torch
from cswinunetr import CSWinUNETR

# 2D binary segmentation (e.g. 3-channel RGB images, background + foreground)
model = CSWinUNETR(in_channels=3, out_channels=2, spatial_dims=2).eval()
with torch.inference_mode():
    logits = model(torch.randn(1, 3, 256, 256))   # N, C, H, W
    prediction = logits.argmax(dim=1)              # N, H, W with class IDs

# 3D multiclass segmentation (e.g. CT with 13 organs + background)
model_3d = CSWinUNETR(in_channels=1, out_channels=14, spatial_dims=3)
# input shape: N, C, D, H, W
```

### Loading a trained FFHQ checkpoint yourself

```python
import torch
from cswinunetr import CSWinUNETR

saved = torch.load("runs/ffhq/best.pt", map_location="cpu", weights_only=True)
model = CSWinUNETR(**saved["model_args"])
model.load_state_dict(saved["model"])
model.eval()
print("Trained to epoch", saved["epoch"], "with val Dice", saved["val_dice"])
```

### Input rules

- Shape: `NCHW` (2D) or `NCDHW` (3D).
- **Every spatial dimension must be a multiple of 32 and at least 64.**
- `out_channels` counts the background, so binary segmentation uses `2`.
- The output contains raw logits. Use `argmax(dim=1)` for labels, or `softmax(dim=1)` for probabilities.
- For large 3D volumes, use sliding-window inference, for example `monai.inferers.sliding_window_inference`.

---

## 12. Troubleshooting

| Error / symptom | Cause | Fix |
|---|---|---|
| `requires-python >=3.10` / syntax errors on install | Python is too old | Create the venv with `python3.11` |
| `FileNotFoundError: .../masked_face_images` | Wrong `--data-root` or folder name | Point to the folder that *contains* the three folders; check the spelling |
| `Missing images for N IDs: ...` | Some split IDs have no file | Re-check the dataset download and preprocessing; run the check script in [Section 6](#6-checking-the-dataset-before-training) |
| `Duplicate image ID ...` | The same filename appears twice in a folder tree | Remove the duplicate copy |
| `Expected aligned 1024x1024 inputs` | Image size does not match `--image-size` | Resize the data or change `--image-size` |
| `Image/texture size mismatch` | The texture map has a different size from the image | Regenerate or resize the texture maps |
| `output-dir must be empty` | The folder holds a previous run | Use a new `--output-dir`, or add `--resume` |
| `Cannot resume with a different ...` | An argument changed between runs | Use exactly the original arguments |
| `CUDA out of memory` | Batch too large | Lower `--batch-size`, add `--use-checkpoint` |
| `Spatial dimensions must be multiples of 32` | Bad input size | Pad or resize to a multiple of 32 |
| `TypeError ... alpha_affine` | `albumentations` is too new | `pip install albumentations==1.3.1` |
| Prediction PNGs look all black | The values are 0/1, not 0/255 | Multiply by 255 to view ([Section 8](#8-inference-and-evaluation)) |
| `all-metrics requires mask-dir` | Metrics need ground truth | Add `--mask-dir` |
| `ModuleNotFoundError: scipy / skimage` | Eval extras not installed | `pip install -e '.[eval]'` |
| Training is extremely slow | Running on CPU/MPS at 1024×1024 | Use a CUDA GPU |
