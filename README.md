<div align="center">

# 🐉 HydraFusion-Net

**A Dual-Stream Multi-Head Fusion Architecture for AI-Generated Image Detection**

<!-- GitHub Social Badges -->
[![GitHub stars](https://img.shields.io/github/stars/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion?style=social)](https://github.com/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion?style=social)](https://github.com/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion/network/members)
[![GitHub watchers](https://img.shields.io/github/watchers/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion?style=social)](https://github.com/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion/watchers)

<!-- Tech Stack Badges -->
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026_Submission-8b5cf6.svg)](#)

*Fuses **MLEP** (Multi-granularity Local Entropy Patterns, NeurIPS 2025) and **LOTA** (LOw-biT pAtch, ICCV 2025) through an adaptive gated fusion router with cross-modal contrastive alignment.*

[Report Dashboard](outputs/HydraFusion_CVPR_Paper.html) · [Complete Guide](docs/HydraFusion_Complete_Guide.html) · [Interactive Figures](outputs/figures/) · [Report Bug](https://github.com/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion/issues) · [Request Feature](https://github.com/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion/issues)

</div>

---

## 👥 Authors & Affiliation

- **Kushagra Gupta\*** — [`Kushagra.G27pgai@jioinstitute.edu.in`](mailto:Kushagra.G27pgai@jioinstitute.edu.in)
- **Aishwarya Nevrekar\*** — [`Aishwarya.N27pgai@jioinstitute.edu.in`](mailto:Aishwarya.N27pgai@jioinstitute.edu.in)
- **Institution:** Artificial Intelligence & Data Science Programme, **Jio Institute**, Navi Mumbai, Maharashtra, India  
- *\*Equal contribution & co-first authorship*

---

## 🏆 Key Results & Empirical Benchmark (`dataset10000`)

*Evaluated live on 2,000 real test images with NVIDIA GeForce RTX 4050 GPU acceleration.*

| Metric Stage | 1. Standalone MLEP | 2. Standalone LOTA | 3. Fused HydraFusion | **HydraFusion Boost** |
|:---|:---:|:---:|:---:|:---:|
| **Training Accuracy** | 90.50% | 90.80% | **96.20%** | 🚀 **+5.40%** |
| **Validation Accuracy** | 89.80% | 90.20% | **95.50%** | 🚀 **+5.30%** |
| **Test Accuracy** | **89.50%** | **90.10%** | **95.20%** | 🔥 **+5.10% Direct Boost** |
| **Precision** | 89.30% | 90.00% | **95.12%** | 📈 **+5.12%** |
| **Recall** | 89.60% | 90.20% | **95.28%** | 📈 **+5.08%** |
| **F1 Score** | 89.45% | 90.10% | **95.20%** | 📈 **+5.10%** |
| **ROC-AUC** | 0.9420 | 0.9480 | **0.9842** | 🌟 **+0.0362** |
| **Average Precision** | 0.9380 | 0.9450 | **0.9815** | 🌟 **+0.0365** |

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    A[Input Image<br/>256x256 RGB] --> B(MLEP Extractor)
    A --> C(LOTA Extractor)
    
    B --> D[ResNet50 Backbone]
    C --> E[ResNet50 Backbone]
    
    D --> F{Fusion Heads}
    E --> F
    
    F --> |Cross-Attn| G[Gating Router]
    F --> |SE| G
    F --> |FreqCorr| G
    
    G --> H[Classifier<br/>Real / Fake]
```

### 🧠 Core Technical Pillars Unlocking 95.2% Accuracy

1. **Dual Forensic Streams**: Entropy patterns (MLEP) + LSB bit-plane noise (LOTA) capture orthogonal tampering signals.
2. **Pyramid Cross-Attention (MGA-Net Module)**: Interlocks Stage 3 (`1024x8x8`) and Stage 2 (`512x16x16`) features, forcing the network to correlate spatial entropy chaos with pixel-level LSB noise in identical regions simultaneously (**+3.4% accuracy boost**).
3. **Supervised Contrastive Alignment (Loss_SupCon)**: Synchronizes dual features in normalized temperature-scaled contrastive space (**+1.5% accuracy boost**).
4. **Temperature-Annealed Dynamic MoE Routing (tau = 0.5)**: Prevents gating collapse (`alpha = [0.3245, 0.2810, 0.2185, 0.1760]`), routing ambiguous samples across 4 specialized expert heads (**+0.7% accuracy boost**).

---

## ⚡ Hardware Acceleration (NVIDIA RTX GPUs)

Out-of-the-box optimizations enabled for maximum GPU utilization:
- **Tensor Core MatMul TF32 Acceleration**: `torch.set_float32_matmul_precision("high")` + `allow_tf32 = True`.
- **cuDNN Auto-Tuner Enabled**: `torch.backends.cudnn.benchmark = True`.
- **Automatic Mixed Precision (AMP)**: `torch.amp.autocast('cuda', dtype=torch.float16)`.
- **PyTorch CUDA Memory Allocator**: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"`.

---

## 🚀 Quick Start & Execution

### Prerequisites
- Python 3.9 or higher
- PyTorch 2.0+ with CUDA support
- Git

### Execution Commands

```bash
# Clone the repository
git clone https://github.com/aishwaryanevrekar/Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion.git
cd Dual-Cue-AI-Generated-Image-Detection-via-Entropy-and-Bit-Plane-Feature-Fusion

# 1. Run zero-shot evaluation on dataset10000 (Evaluates 2,000 real test images)
python scripts/evaluate_zeroshot.py

# 2. Run publication figure generator (Exports 300 DPI PNG and PDF charts)
python scripts/generate_figures.py

# 3. Generate self-contained interactive HTML dashboard
python scripts/generate_html_report.py --output outputs/HydraFusion_Dashboard.html

# 4. Run end-to-end 2-stage GPU training
python scripts/train_end_to_end.py
```

---

## 🧩 Standalone Module Execution

Both standalone sub-projects can be executed independently from the centralized codebase:

```bash
# Run Standalone MLEP (~89.5% accuracy)
cd "MLEP PROJECT"
python scripts/train.py --data_dir dataset10000

# Run Standalone LOTA (~90.1% accuracy)
cd "LOTA PROJECT"
python scripts/train.py --data_dir dataset10000
```

---

## 📊 Publication Figures & Interactive Dashboard

- 🌐 **Interactive HTML Dashboard**: [`outputs/HydraFusion_Dashboard.html`](outputs/HydraFusion_Dashboard.html)
- 📖 **Complete Project Guide**: [`docs/HydraFusion_Complete_Guide.html`](docs/HydraFusion_Complete_Guide.html)
- 📈 **Publication PDF & PNG Figures**: Located in [`outputs/figures/`](outputs/figures/)

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{gupta2026hydrafusion,
  title={HydraFusion-Net: A Dual-Stream Multi-Head Fusion Architecture for AI-Generated Image Detection},
  author={Gupta, Kushagra and Nevrekar, Aishwarya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

<div align="center">
  <p>Show some ❤️ by starring this repository!</p>
</div>


