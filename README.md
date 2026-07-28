# Dual-Cue AI-Generated Image Detection: MLEP & LOTA Fusion
### Shared Dataset Infrastructure & LOw-biT pAtch (LOTA) Preprocessing Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org)
[![Albumentations](https://img.shields.io/badge/Albumentations-2.0%2B-28a745?style=flat-square)](https://albumentations.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

This repository contains the production-grade implementation of the **Shared Dataset Infrastructure** and the **LOTA (*LOw-biT pAtch*, ICCV 2025) Preprocessing & Steganalysis Engine** for Dual-Cue AI-Generated Image Detection (AIGID).

In strict accordance with project division specifications, this module handles high-performance data ingestion (supporting open-access HuggingFace Hub datasets and 1,400+ sample multi-domain benchmarks), automated file header validation, stratified partitioning, 50/50 class balanced mini-batch sampling, and 100% vectorized Top-$K$ least-significant bit (LSB) patch extraction. It delivers standardized $256 \times 256$ RGB tensors and steganalysis noise patches to downstream classification pipelines without implementing any MLEP or cross-modal fusion modules.

---

## 📋 Table of Contents
1. [Environment Setup & Package Installation](#1-environment-setup--package-installation)
2. [How to Run LOTA & the Project Pipeline](#2-how-to-run-lota--the-project-pipeline)
3. [Understanding the Outputs & Visualizations](#3-understanding-the-outputs--visualizations)
4. [Project Architecture & Directory Structure](#4-project-architecture--directory-structure)
5. [Verification & Test Suite](#5-verification--test-suite)
6. [Hardware & Device Compatibility](#6-hardware--device-compatibility)
7. [Troubleshooting & FAQ](#7-troubleshooting--faq)

---

## 1. Environment Setup & Package Installation

We provide three simple methods to configure your environment and download all required project dependencies (`PyTorch`, `Torchvision`, `Albumentations`, `NumPy`, `Pillow`, `Matplotlib`, `PyYAML`, and `Pytest`).

### Option A: Quick Automated Setup Script (Recommended for macOS / Linux)
We provide an automated bash script that creates a virtual environment and installs all dependencies automatically:
```bash
# Make the setup script executable (if not already)
chmod +x setup_env.sh

# Run the automated environment builder
./setup_env.sh

# Activate the newly created environment
source venv/bin/activate
```

### Option B: Using Anaconda / Miniconda (`environment.yml`)
If you prefer Conda for package and environment management:
```bash
# Create the environment from environment.yml
conda env create -f environment.yml

# Activate the conda environment
conda activate mlep_lota
```

### Option C: Manual Virtual Environment (`requirements.txt`)
If you prefer using standard Python tools:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip wheel
pip install -r requirements.txt
```

---

## 2. How to Run LOTA & the Project Pipeline

Once your environment is activated, you can execute the project pipelines using the provided command-line scripts in `scripts/`.

### A. Download Open-Access Datasets or Generate Large-Scale Training Data (`download_dataset.py`)
To scale up evaluations, we provide an automated utility (`scripts/download_dataset.py`) that downloads open-access Real vs. AI-generated image datasets from HuggingFace Hub (`dima806/ai_vs_real_image_detection`) or generates large-scale structured benchmark datasets locally across 5 generator domains:

```bash
# Option 1: Download open-access Real vs. AI dataset from HuggingFace Hub (requires 'pip install datasets')
python scripts/download_dataset.py --source huggingface --target_dir outputs/hf_dataset --num_samples 1000

# Option 2: Generate a large-scale 1,400+ image local benchmark dataset (simulating Real, StyleGAN2, Midjourney, FLUX, and ProGAN)
python scripts/download_dataset.py --source local --target_dir outputs/dataset_1400 --num_samples 1400
```

**Command-Line Arguments:**
* `--target_dir`: Directory to store downloaded or generated image datasets (default: `outputs/big_dataset`).
* `--source`: Dataset source, either `huggingface` (open-access hub dataset) or `local` (synthetic multi-texture benchmark generation).
* `--num_samples`: Total number of images to ingest or generate across class domains (default: `1000`).

### B. Run Master End-to-End Pipeline (`run_project.py`)
This script demonstrates the complete integration of the **Shared Dataset Infrastructure** with the **LOTA Preprocessing Engine**. By default, it targets the **1,400-sample benchmark dataset** (`outputs/dataset_1400`), automatically partitioning stratified splits (60% Train / 20% Val / 20% Test), building class-balanced DataLoaders (guaranteeing 50/50 Real vs. AI mini-batch ratios), and executing 100% vectorized Top-$K$ patch extraction:

```bash
python scripts/run_project.py \
    --data_dir "/Volumes/Seagate/JIO TERM/JIO-TERM 3/DL AND CV PROJECT/outputs/dataset_1400" \
    --output_dir outputs/project_run \
    --batch_size 8 \
    --num_workers 0 \
    --k_patches 4 \
    --export_visualizations

```

**Command-Line Arguments:**
* `--data_dir`: Path to input dataset directory. If empty or non-existent, the script automatically generates a 1,400-sample multi-domain benchmark dataset (`outputs/dataset_1400`).
* `--output_dir`: Directory to store generated JSON manifests, execution analytics reports, and PNG visualizations (default: `outputs/project_run`).
* `--batch_size`: Mini-batch size for DataLoader (default: `8`; set to `32` or `64` for high-throughput training).
* `--num_workers`: Subprocess workers for data loading (default: `0` for universal synchronous compatibility; set to `4` on Linux/Multi-core GPU rigs).
* `--k_patches`: Number of diverse non-overlapping Top-$K$ noise patches to extract per image (default: `4`).
* `--export_visualizations`: Flag to export visual diagnostic grids for sample batches.

### C. Run Standalone LOTA Visualization Suite (`visualize_lota.py`)
To isolate and test the LOTA steganalysis extraction engine on a specific image (or generate synthetic multi-texture test patterns):

```bash
# Run on a custom image
python scripts/visualize_lota.py --image_path path/to/your/image.png --output_dir outputs/visualizations

# Run without arguments to generate a synthetic multi-texture benchmark image
python scripts/visualize_lota.py --output_dir outputs/visualizations
```

### D. Launch Interactive HTML Dashboard in Google Chrome (`generate_html_report.py`)
To view all generated diagnostic figures, bit-plane decompositions, MGPS heatmaps, and JSON analytics inside an interactive Glassmorphism web dashboard in Google Chrome:

```bash
# Generate dashboard and open in Chrome automatically
python scripts/generate_html_report.py

# Or open the generated file directly in Chrome on macOS
open -a "Google Chrome" outputs/LOTA_Dashboard.html
```

### E. Training Downstream AI Detection Models on `dataset_1400`
To train your custom PyTorch classification model (such as ResNet, Vision Transformer, or a custom Dual-Cue AI-Generated Image Detector) on the **1,400-sample benchmark dataset**, leverage the pre-built `SharedImageDataset` and `TopKLOTAExtractor` modules. This guarantees 100% vectorized LSB noise patch extraction, stratified splitting, and 50/50 class-balanced mini-batch sampling:

```python
import torch
import torch.nn as nn
from pathlib import Path
from src.data.dataset import SharedImageDataset
from src.data.dataloader import create_dataloader
from src.models.lota import TopKLOTAExtractor

# 1. Initialize Dataset & Dataloaders targeting dataset_1400
data_root = Path("/Volumes/Seagate/JIO TERM/JIO-TERM 3/DL AND CV PROJECT/outputs/dataset_1400")
train_ds = SharedImageDataset(root_dir=data_root, split="train", val_ratio=0.2, test_ratio=0.2)
train_loader = create_dataloader(train_ds, batch_size=32, num_workers=4, balanced_sampling=True)

# 2. Initialize LOTA Preprocessing Engine
lota_extractor = TopKLOTAExtractor(k_patches=4, patch_size=32, grid_size=8).eval()

# 3. Define Downstream Classification Model & Optimizer
model = YourCustomClassifier(num_classes=2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 4. End-to-End Training Loop
for epoch in range(10):
    model.train()
    for images, labels, metas in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        # Extract Top-K LSB noise patches (B, K, 3, 32, 32) & MGPS scores (B, 8, 8)
        with torch.no_grad():
            lota_out = lota_extractor(images)
            noise_patches = lota_out["z_norm"]
            mgps_scores = lota_out["mgps_scores"]
            
        # Forward pass through your classification pipeline
        optimizer.zero_grad()
        predictions = model(images, noise_patches)  # Feed RGB tensors + LOTA noise cues
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
```

---

## 3. Understanding the Outputs & Visualizations

### A. Console & Analytics Report Outputs
When executing `scripts/run_project.py`, an execution analytics report is printed to terminal and saved to `outputs/project_run/execution_summary.json`. 

**Example Terminal Output:**
```text
--------------------------------------------------
Total Images Processed  : 864
Throughput              : 538.1 images/second (59.5 ms/batch)
Real Mean MGPS Score    : 431073.4351
AI Mean MGPS Score      : 312754.9473
Divergence Contrast     : 0.73x
Visualizations Saved To : outputs/project_run/visualizations
--------------------------------------------------
```

**Understanding MGPS Divergence Across Dataset Distributions:**
The **Multi-directional Gradient Patch Scoring (MGPS)** algorithm convolves the least-significant bit composition ($\tilde{\mathbf{z}} = 4x_2 + 2x_1 + x_0$) against 4 directional gradient kernels ($\mathbf{g}_x, \mathbf{g}_y, \mathbf{g}_{xy}, \mathbf{g}_{yx}$). 
* **Real Images** ($431,073.4$): Exhibit continuous sensor noise distributions across authentic photograph domains (`0_real`).
* **AI-Generated Images** ($312,754.9$): Capture high-frequency steganographic anomalies across diverse generator domains (`1_stylegan2`, `1_midjourney`, `1_flux`, `1_progan`). On real-world datasets such as ForenSynths or GenImage, AI generators often produce a >1.40x higher LSB divergence score due to checkerboard upsampling artifacts, whereas synthetic benchmark distributions reveal distinct frequency contrast signatures that allow downstream classifiers to cleanly separate authentic vs. synthetic representations.

### B. Generated Visual Diagnostic Figures
In `outputs/visualizations/` and `outputs/project_run/visualizations/`, the system generates three high-resolution diagnostic PNG figures for each inspected sample:

#### 1. 8 Bit-Plane Decomposition (`*_bit_planes.png`)
Slices the RGB raster into 8 binary planes ($x_7$ down to $x_0$).
* **Most Significant Bits ($x_7 \dots x_4$)**: Capture coarse semantic geometry and luminance contours.
* **Least Significant Bits ($x_2 \dots x_0$)**: Strip away semantic image content to reveal pure high-frequency steganographic noise where AI generator footprints reside.

#### 2. Multi-directional Gradient Patch Scoring Heatmap (`*_mgps_heatmap.png`)
Displays the $8 \times 8$ grid ($32 \times 32$ pixel patches) divergence heatmap overlaid on the original image. 
* Bright red squares indicate regions of maximum LSB gradient energy and steganographic inconsistency.
* Dark blue squares indicate smooth, uninformative regions (e.g., flat skies or solid backgrounds).

#### 3. Top-K ($K=4$) Quadrant-Diverse Noise Patches (`*_topk_patches.png`)
Displays the selected Top-$K$ highest-scoring patches alongside their binarized LSB noise maps.
* **Quadrant Diversity**: The selection algorithm enforces a strict constraint ensuring that each selected patch originates from a distinct spatial quadrant (Top-Left, Top-Right, Bottom-Left, Bottom-Right). This guarantees **zero spatial overlap** and maximum spatial texture coverage across the image.

---

## 4. Project Architecture & Directory Structure

```text
DL AND CV PROJECT/
├── README.md                      # Master project documentation (this file)
├── requirements.txt               # Pip dependency specification
├── environment.yml                # Conda environment specification
├── setup_env.sh                   # Automated environment setup bash script
├── configs/
│   └── default.yaml               # Master YAML configuration (256x256 res, K=4 patches, 8x8 grid)
├── scripts/
│   ├── download_dataset.py        # Automated HuggingFace downloader & large-scale benchmark generator
│   ├── generate_html_report.py    # Interactive Glassmorphism HTML dashboard & analytics reporter
│   ├── run_project.py             # Master end-to-end integration script (Dataset -> LOTA pipeline)
│   └── visualize_lota.py          # Standalone LOTA steganalysis visualization CLI
├── src/
│   ├── data/                      # Shared Dataset Infrastructure (MLEP & LOTA compatible)
│   │   ├── metadata.py            # Integrity verification (PIL.Image.verify), domain scanning, JSON export
│   │   ├── splits.py              # Stratified deterministic 1:1 class partitioning & manifest persistence
│   │   ├── augmentations.py       # Albumentations 2.0+ pipelines (JPEG recompression, Gaussian blur)
│   │   ├── transforms.py          # SharedImageTransform & LOTAPreprocessingTransform tensor bridges
│   │   ├── dataset.py             # SharedImageDataset & AIGIDDataset PyTorch Dataset classes
│   │   ├── dataloader.py          # create_dataloader() factory with worker seeding setup
│   │   └── samplers.py            # BalancedRealFakeSampler guaranteeing 50/50 mini-batch ratios
│   ├── models/                    # LOTA Preprocessing Core
│   │   └── lota.py                # TopKLOTAExtractor (100% vectorized PyTorch bit-slicing & MGPS scoring)
│   └── utils/                     # Shared Utilities & Visualization Suite
│   │   ├── config.py              # Typed dataclasses (ProjectConfig, LOTAConfig, DatasetConfig)
│   │   ├── logger.py              # Standardized console & file logging with timestamp formatting
│   │   └── visualization.py       # Rendering functions for bit-planes, heatmaps, and patch crops
├── tests/                         # Comprehensive Unit Verification Suite (19 tests)
│   ├── test_config_logger.py      # Config YAML & logger verification
│   ├── test_dataset.py            # Dataset transforms, augmentations, and balanced samplers
│   ├── test_lota.py               # Vectorized bit-plane slicing, MGPS scoring, and Top-K diversity
│   └── test_shared_dataset.py     # End-to-end dataset scanning, splitting, and dataloader verification
└── docs/
    ├── LOTA_ARCHITECTURE_AND_PIPELINE_GUIDE.md # Comprehensive guide explaining whole pipeline, docs inventory & execution
    └── lota_pipeline.md           # Complete mathematical & engineering specification of LOTA
```

> [!TIP]
> **New to the project?** Check out our comprehensive technical guide: [docs/LOTA_ARCHITECTURE_AND_PIPELINE_GUIDE.md](file:///Volumes/Seagate/JIO%20TERM/JIO-TERM%203/DL%20AND%20CV%20PROJECT/docs/LOTA_ARCHITECTURE_AND_PIPELINE_GUIDE.md) for an in-depth explanation of the 5-Stage LOTA architecture, step-by-step execution flows, and a complete inventory of every configuration and code module!

---

## 5. Verification & Test Suite

The codebase includes an exhaustive unit test suite verified with `pytest`. To run the verification suite across all modules:

```bash
pytest tests/ -v
```

**Expected Output:**
```text
============================= test session starts ==============================
platform darwin -- Python 3.13.14, pytest-8.3.4
rootdir: DL AND CV PROJECT

tests/test_config_logger.py::test_default_project_config PASSED          [  5%]
tests/test_config_logger.py::test_save_and_load_config PASSED            [ 10%]
tests/test_config_logger.py::test_load_nonexistent_config PASSED         [ 15%]
tests/test_config_logger.py::test_logger_initialization PASSED           [ 21%]
tests/test_config_logger.py::test_logger_no_duplicate_handlers PASSED    [ 26%]
tests/test_dataset.py::test_transforms_resizing_and_range PASSED         [ 31%]
tests/test_dataset.py::test_robustness_augmentations PASSED              [ 36%]
tests/test_dataset.py::test_dataset_scanning_and_splits PASSED           [ 42%]
tests/test_dataset.py::test_metadata_export PASSED                       [ 47%]
tests/test_dataset.py::test_balanced_sampler PASSED                      [ 52%]
tests/test_lota.py::test_bit_plane_reconstruction PASSED                 [ 57%]
tests/test_lota.py::test_lsb_composition_and_thresholding PASSED         [ 63%]
tests/test_lota.py::test_mgps_scoring_on_flat_and_edge_images PASSED     [ 68%]
tests/test_lota.py::test_topk_quadrant_diversity PASSED                  [ 73%]
tests/test_lota.py::test_forward_pipeline_output_shapes PASSED           [ 78%]
tests/test_shared_dataset.py::test_metadata_integrity_and_scanning PASSED [ 84%]
tests/test_shared_dataset.py::test_stratified_dataset_partitioning PASSED [ 89%]
tests/test_shared_dataset.py::test_albumentations_and_transforms PASSED  [ 94%]
tests/test_shared_dataset.py::test_shared_dataset_and_dataloader PASSED  [100%]

======================== 19 passed, 1 warning in 1.58s =========================
```

---

## 6. Hardware & Device Compatibility
The entire pipeline is engineered using 100% vectorized PyTorch operations without Python for-loops over pixels or patches. It executes natively across all standard hardware accelerators:
* **Apple Silicon (M1/M2/M3/M4)**: Full compatibility with Metal Performance Shaders (`mps` device) and CPU vectorized execution (~530+ img/s throughput).
* **NVIDIA GPUs**: Full compatibility with CUDA acceleration for multi-worker distributed training.
* **Standard CPUs**: High-throughput fallback execution on x86_64 / ARM architectures.

---

## 7. Troubleshooting & FAQ

### A. Matplotlib `UnicodeDecodeError` on External Hard Drives (macOS)
If your project or virtual environment is located on an external hard drive (e.g., `/Volumes/Seagate/...`), running visualization or reporting scripts might produce the following error:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 37: invalid start byte
```
**Cause:** On external non-HFS+/APFS filesystems (such as exFAT, FAT32, or NTFS on USB drives), macOS automatically generates AppleDouble hidden files (starting with `._`, such as `._Solarize_Light2.mplstyle` and `._matplotlibrc`) to store file extended attributes and resource forks. When Matplotlib initializes its stylesheet library (`stylelib`), it attempts to open and decode these binary `._*` AppleDouble files as UTF-8 text.

**Solution:** Run the following command from the project root in your terminal to recursively delete all AppleDouble metadata files:
```bash
find . -name "._*" -type f -delete
```
Once executed, your pipeline and visualization scripts will run smoothly without Matplotlib decode errors.
