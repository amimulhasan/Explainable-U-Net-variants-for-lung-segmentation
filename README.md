# 🫁 Explainable U-Net Variants for Lung Segmentation on Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

Deep learning project for **lung segmentation** from chest X-ray images using three U-Net variants, with **K-fold cross-validation** and **Grad-CAM** explainability.

Inspired by: *Abushahla & Arslan (2026), Tuberk Toraks 74(2):84–97*

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Task** | Semantic segmentation (lung masks) |
| **Models** | Baseline U-Net · Attention U-Net · Shallow U-Net |
| **Input size** | 256 × 256 × 3 |
| **Validation** | K-Fold (default k=5) |
| **Metrics** | Dice, IoU (Jaccard), Precision, Recall |
| **XAI** | Grad-CAM visualizations |
| **Framework** | TensorFlow / Keras |

---

## 📁 Project Structure

```
lung-segmentation/
├── lung-cancer-segmentation.ipynb   # Full pipeline notebook
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/                            # You add this
│   ├── images/                      # Chest X-ray images
│   └── masks/                       # Corresponding lung masks
└── outputs/                         # Created at runtime
    ├── *.keras                      # Trained models
    ├── model_comparison_summary.csv
    ├── qualitative_comparison.png
    ├── gradcam_comparison.png
    └── eda_samples.png
```

---

## 📥 Dataset Setup

Images and masks are **not** included (medical data size / licensing).

### Option A — Kaggle dataset
If you use the dataset  
`amimulahasanrofik/chest-x-ray-segmentation`:

1. Add the dataset in a Kaggle notebook, **or**
2. Download and place files as:

```text
data/images/   ← X-ray images (.png / .jpg)
data/masks/    ← binary lung masks (same filenames)
```

### Option B — Environment variables
```bash
export IMAGE_DIR=/path/to/images
export MASK_DIR=/path/to/masks
export OUTPUT_DIR=./outputs
```

The notebook auto-matches image–mask pairs by filename stem.

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/lung-segmentation.git
cd lung-segmentation

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# Place data under data/images and data/masks, then:
jupyter notebook lung-cancer-segmentation.ipynb
```

### Recommended settings (edit in notebook)

| Parameter | Default | Notes |
|-----------|---------|--------|
| `IMG_SIZE` | 256 | Paper setting |
| `BATCH_SIZE` | 8 | Lower if OOM |
| `EPOCHS` | 50 | Paper used 100 |
| `N_FOLDS` | 5 | Use 2–3 for quick tests |

**GPU strongly recommended** (Kaggle / Colab / local CUDA).

---

## 🔬 Pipeline (A–Z)

1. **Pair matching** — Match images & masks by filename  
2. **EDA** — Sample overlays saved to `outputs/eda_samples.png`  
3. **Preprocessing** — Resize 256×256, normalize images [0,1], binarize masks  
4. **Models**
   - Baseline U-Net  
   - Attention U-Net (attention gates)  
   - Shallow U-Net (lighter capacity)  
5. **Training** — K-fold CV, Dice + BCE style objectives  
6. **Evaluation** — Dice, IoU, precision, recall per fold & overall  
7. **Qualitative results** — Side-by-side predictions  
8. **Grad-CAM** — Explainable heatmaps for model decisions  
9. **Exports** — CSV summaries, PNG figures, `.keras` weights  

---

## 📈 Expected Outputs

| File | Description |
|------|-------------|
| `model_comparison_summary.csv` | Table-style metrics comparison |
| `qualitative_comparison.png` | Image / GT / predictions |
| `gradcam_comparison.png` | Grad-CAM overlays |
| `per_sample_metrics_<model>.csv` | Per-image Dice / IoU |
| `<model>.keras` | Saved trained weights |

---

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow 2.x / Keras  
- OpenCV, NumPy, Pandas, Matplotlib  
- scikit-learn (KFold)

---

## 📝 Citation / Reference

If you use this pipeline in academic work, please also cite the original study that motivated the architecture comparison:

> Abushahla & Arslan (2026). Explainable U-Net variants for lung segmentation on chest X-rays. *Tuberk Toraks*, 74(2), 84–97.

---

## 📄 License

MIT License — free to use and modify.

---

**Made with ❤️ for learning Medical Image Segmentation & XAI**
