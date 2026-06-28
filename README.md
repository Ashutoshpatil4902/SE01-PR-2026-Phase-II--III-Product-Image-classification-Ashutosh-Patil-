# 🛒 E-Commerce Product Image Classification using CNN-Based Deep Learning

> Classifying e-commerce product images into **9 categories** — Grocery, Electronics, Clothing, Home/Kitchen, Beauty, Sports, Baby Products, Hobby, and Pet Supplies — using custom and transfer-learning CNN models with Grad-CAM and SHAP explainability.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red?logo=keras)](https://keras.io/)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/patilashutosh09/final-cnn-file)
[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://ashutosh4902-product-image-classifier.hf.space)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Architectures](#-model-architectures)
- [Results](#-results)
- [Explainability (Grad-CAM & SHAP)](#-explainability-grad-cam--shap)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Live Demo](#-live-demo)
- [References](#-references)

---

## 🔍 Project Overview

E-commerce platforms list millions of products every day, and manually categorising every image into the correct product class simply doesn't scale. This project investigates whether **Convolutional Neural Networks (CNNs)** can learn to distinguish product categories from image content alone — packaging shape, colour, texture, logo, and material — and produce predictions that are both accurate and interpretable.

This is a **9-class image classification** problem:

| Label | Category |
|-------|----------|
| `GROCERY` | Packaged food, beverages, and snacks |
| `HOME_KITCHEN_TOOLS` | Cookware, appliances, and cleaning products |
| `ELECTRONICS` | Cables, headphones, displays, and devices |
| `PET_SUPPLIES` | Pet food, accessories, and care products |
| `SPORTS_OUTDOOR` | Exercise gear, outdoor equipment, and athletic wear |
| `BEAUTY_HEALTH` | Cosmetics, personal care, and supplements |
| `HOBBY_ARTS_STATIONERY` | Craft supplies, instruments, and office materials |
| `BABY_PRODUCTS` | Diapers, baby food, and feeding accessories |
| `CLOTHING_ACCESSORIES_JEWELLERY` | Apparel, footwear, bags, and jewellery |

Three models are trained and compared:

- ✅ **Custom CNN** (built from scratch as a baseline)
- ✅ **MobileNetV2** (transfer learning + fine-tuning)
- ✅ **ResNet50** (transfer learning + fine-tuning)

Model decisions are explained visually using **Grad-CAM** (class activation heatmaps) and **SHAP** (pixel-level feature attribution).

---

## 📦 Dataset

**Source:** [E-commerce Product Images 18K — Fatih Kögüş (Kaggle)](https://www.kaggle.com/datasets/fatihkgg/ecommerce-product-images-18k)

Product images are organised into per-category subfolders. Each image is an RGB photograph captured under real-world e-commerce conditions — varying lighting, backgrounds, and seller-uploaded quality — making this a genuinely challenging classification task.

### Dataset Summary

| Attribute | Details |
|-----------|---------|
| Total Images Used | 18,175 |
| Number of Classes | 9 |
| Image Format | JPEG / PNG (RGB) |
| Input Resolution (after resize) | 224 × 224 pixels |
| Class Balance | Imbalanced (ratio ≈ 3.7 : 1) |

### Class Distribution

| Category | Images | % of Total |
|----------|--------|-----------|
| GROCERY | 5,166 | 28.4% |
| HOME_KITCHEN_TOOLS | 2,228 | 12.3% |
| ELECTRONICS | 1,757 | 9.7% |
| PET_SUPPLIES | 1,637 | 9.0% |
| SPORTS_OUTDOOR | 1,605 | 8.8% |
| BEAUTY_HEALTH | 1,562 | 8.6% |
| HOBBY_ARTS_STATIONERY | 1,417 | 7.8% |
| BABY_PRODUCTS | 1,412 | 7.8% |
| CLOTHING_ACCESSORIES_JEWELLERY | 1,391 | 7.7% |

> **Class imbalance:** GROCERY contains approximately 3.7× more images than the smallest class. Stratified splitting is used at every stage to preserve proportions across train, validation, and test sets.

### Train / Validation / Test Split (70 / 15 / 15)

| Split | Images | Proportion |
|-------|--------|-----------|
| Train | 12,722 | 70% |
| Validation | 2,726 | 15% |
| Test | 2,727 | 15% |

> Splits were stratified to ensure each class is proportionally represented in all three subsets.

---

## 🔧 Methodology

### 1. Image Preprocessing

All images are resized to 224 × 224 pixels using bilinear interpolation and pixel values are normalised to the [0, 1] range. RGB colour channels are retained because colour is a primary discriminating signal between product categories.

### 2. Data Augmentation (Training Only)

Augmentations are applied on-the-fly using Keras augmentation layers to reduce overfitting:

| Transformation | Range / Setting |
|----------------|----------------|
| Random horizontal flip | — |
| Random rotation | ±5° |
| Random zoom | ±10% |
| Random contrast adjustment | ±10% |

> Vertical flipping and large rotations are deliberately excluded — they produce unrealistic product orientations inconsistent with e-commerce photography conventions.

### 3. Model Training Strategy

Transfer-learning models use a **two-stage fine-tuning strategy**:

- **Phase 1 (Head Training):** Pre-trained backbone frozen; only the new classification head is trained for 5 epochs
- **Phase 2 (Fine-Tuning):** Last 30–40 backbone layers unfrozen; full network trained at reduced learning rate for 3 epochs

### 4. Explainability

Model decisions are inspected using:
- **Grad-CAM** — class-discriminative spatial heatmaps from the last convolutional layer
- **SHAP** — pixel-level attribution using a partition-based image explainer with superpixel masking

---

## 🏗️ Model Architectures

### Custom CNN (Built from Scratch)

| Layer / Block | Configuration | Output Shape |
|---------------|--------------|--------------|
| Input | 224 × 224 × 3 RGB | 224 × 224 × 3 |
| Rescaling | Normalize to [0, 1] | 224 × 224 × 3 |
| Block 1: Conv2D | 32 filters, 3×3, ReLU, BatchNorm | 224 × 224 × 32 |
| Block 1: MaxPool + Dropout | 2×2 pool | 112 × 112 × 32 |
| Block 2: Conv2D | 64 filters, 3×3, ReLU, BatchNorm | 112 × 112 × 64 |
| Block 2: MaxPool + Dropout | 2×2 pool | 56 × 56 × 64 |
| Block 3: Conv2D | 128 filters, 3×3, ReLU, BatchNorm | 56 × 56 × 128 |
| Block 3: MaxPool + Dropout | 2×2 pool | 28 × 28 × 128 |
| GlobalAveragePooling2D | — | 128 |
| Dense + Dropout | 256 neurons, ReLU | 256 |
| Output (Softmax) | 9 classes | 9 |

> **Total Parameters:** 457,961 &nbsp;|&nbsp; **Optimizer:** Adam &nbsp;|&nbsp; **Loss:** Sparse Categorical Cross-Entropy &nbsp;|&nbsp; **Grad-CAM Layer:** `custom_last_relu`

### Transfer Learning Models

| Model | Backbone | Pre-training | Fine-tuned Layers | Total Parameters | Grad-CAM Layer |
|-------|----------|-------------|-------------------|-----------------|----------------|
| MobileNetV2 | MobileNetV2 | ImageNet | Last 30 layers | 2,269,513 | `out_relu` |
| ResNet50 | ResNet50 | ImageNet | Last 40 layers | 23,606,153 | `conv5_block3_out` |

Both transfer models use the same classification head appended to the backbone:
`GlobalAveragePooling2D → Dense(128/256, ReLU) → Dropout → Dense(9, Softmax)`

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Initial learning rate | 1e-4 |
| Loss function | Sparse categorical cross-entropy |
| Batch size | 32 |
| Head-training epochs | 5 |
| Fine-tuning epochs | 3 |
| Early stopping patience | 2 epochs (monitor: `val_loss`) |
| LR reduction factor | 0.3 (patience: 1 epoch) |
| Minimum learning rate | 1e-6 |
| Platform | Kaggle (GPU P100) |

---

## 📊 Results

### Final Model Comparison (Test Set — 2,727 images)

| Model | Test Accuracy | Weighted F1 | Precision | Recall | Train Time | Parameters |
|-------|-------------|-------------|-----------|--------|-----------|-----------|
| Custom CNN | 45.47% | 0.3858 | 0.4117 | 0.4547 | ~4.0 min | 457,961 |
| MobileNetV2 | 71.76% | 0.7106 | 0.7113 | 0.7176 | ~4.2 min | 2,269,513 |
| **ResNet50** | **76.93%** | **0.7680** | **0.7739** | **0.7693** | ~6.9 min | 23,606,153 |

> **ResNet50** achieves the best overall accuracy (+26.4 pp over the custom CNN baseline). **MobileNetV2** offers the best accuracy-to-parameter ratio — nearly identical training time to the custom CNN but with a 22-point accuracy advantage.

### Key Observations

- Transfer learning is not optional on a dataset of this scale. The Custom CNN maxed out at ~45% accuracy after 8 epochs, while both transfer-learning models exceeded 70% from the first epoch thanks to ImageNet pre-trained features.
- **ResNet50** is the strongest classifier (76.93% accuracy, 76.80% F1), but it is ~52× larger than the Custom CNN and requires ~70% more training time than MobileNetV2.
- **MobileNetV2** is the practical deployment choice: 71.76% accuracy with only 2.3M parameters and virtually the same training time as the Custom CNN.
- **GROCERY** is the easiest category to classify (≥92% recall across all models), reflecting its distinctive branded packaging.
- **HOME_KITCHEN_TOOLS** is the hardest category — heavily confused with GROCERY, BEAUTY_HEALTH, and ELECTRONICS — because cookware, containers, and small appliances share similar visual form factors.
- The most persistent confusion pairs are: HOME_KITCHEN_TOOLS ↔ GROCERY and SPORTS_OUTDOOR ↔ HOME_KITCHEN_TOOLS.

---

## 🔍 Explainability (Grad-CAM & SHAP)

### Grad-CAM

Grad-CAM heatmaps are generated from the final convolutional layer of each model to visualise which spatial regions drive each prediction:

- ✅ **Correctly classified images** — MobileNetV2 and ResNet50 show focused, product-body-centred attention on packaging, labels, and diagnostically distinctive product features
- ❌ **Misclassified images** — the Custom CNN produces diffuse, double-lobed activation patterns that resemble accessory shapes rather than product-specific features, explaining confusions such as predicting CLOTHING_ACCESSORIES_JEWELLERY for a BABY_PRODUCTS wipe dispenser

### SHAP

SHAP pixel attribution confirms and extends the Grad-CAM findings:

- Transfer-learning models assign tightly concentrated **positive attribution (pink/red)** to the product body and distinctive features, with clear suppression of background regions
- The Custom CNN assigns positive attribution broadly across background and edge regions, with no clear focus on diagnostically meaningful product areas
- Cross-comparison of Grad-CAM and SHAP on the same images shows strong agreement for the transfer-learning models, increasing confidence that predictions are grounded in meaningful visual evidence

---

## 📁 Project Structure

```
SE01-PR-2026-Phase-II--III-Product-Image-classification-Ashutosh-Patil/
│
├── notebooks/
│   └── final-cnn-file.ipynb                          # Main notebook (all models + explainability)
│
├── outputs/
│   ├── figures/
│   │   ├── figure_1_dataset_samples.pdf              # Sample images per class
│   │   ├── figure_1a_class_distribution.pdf          # Class distribution bar chart
│   │   ├── figure_2_model_design_workflow.pdf        # Pipeline workflow diagram
│   │   ├── figure_3_accuracy_and_loss_curves.pdf     # Training / validation curves
│   │   ├── figure_4_confusion_matrix_custom_cnn_normalized.pdf
│   │   ├── figure_4_confusion_matrix_mobilenetv2_normalized.pdf
│   │   ├── figure_4_confusion_matrix_resnet50_normalized.pdf
│   │   ├── figure_5_grad_cam_correct_prediction_example.pdf
│   │   ├── figure_6_grad_cam_misclassification_example.pdf
│   │   ├── figure_7_shap_custom_cnn.pdf
│   │   ├── figure_7_shap_mobilenetv2.pdf
│   │   └── figure_7_shap_resnet50.pdf
│   │
│   └── tables/
│       ├── class_distribution.csv
│       ├── dataset_description.csv
│       ├── model_metrics.csv
│       ├── model_architecture_table.csv
│       ├── training_time_and_parameters.csv
│       ├── train_validation_test_split.csv
│       ├── final_comparison_table.csv
│       ├── custom_cnn_test_predictions.csv
│       ├── mobilenetv2_test_predictions.csv
│       └── resnet50_test_predictions.csv
│
├── report/
│   └── Product_Image_Classification_Report.docx
│
├── README.md
└── requirements.txt
```

---

## 🚀 Setup & Usage

### Requirements

```bash
pip install tensorflow keras scikit-learn numpy pandas matplotlib seaborn shap pillow
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

> Python 3.9+ and TensorFlow 2.x recommended. The notebook was developed and tested on **Kaggle** (GPU P100).

### Option 1 — Run on Kaggle (Recommended)

This project is designed to run on **Kaggle Notebooks** with GPU acceleration — no local setup required.

1. Open the notebook on Kaggle: [final-cnn-file — Kaggle Notebook](https://www.kaggle.com/code/patilashutosh09/final-cnn-file)
2. Attach the [E-commerce Product Images 18K](https://www.kaggle.com/datasets/fatihkgg/ecommerce-product-images-18k) dataset as an input
3. Enable **GPU P100** accelerator under *Settings → Accelerator*
4. Click **Run All** — the notebook auto-detects the dataset path and runs the full pipeline end-to-end

### Option 2 — Run Locally

```bash
git clone https://github.com/Ashutoshpatil4902/SE01-PR-2026-Phase-II--III-Product-Image-classification-Ashutosh-Patil-.git
cd SE01-PR-2026-Phase-II--III-Product-Image-classification-Ashutosh-Patil-
pip install -r requirements.txt
jupyter notebook notebooks/final-cnn-file.ipynb
```

Then update `USER_DATASET_PATH` in the second notebook cell to point to your local dataset directory.

> ⚠️ Local execution requires a CUDA-enabled GPU for reasonable training times. CPU-only runs are possible but will be significantly slower.

---

## 🌐 Live Demo

A live interactive demo is deployed on **Hugging Face Spaces** — upload any product image and get an instant prediction with confidence scores:

👉 **[https://ashutosh4902-product-image-classifier.hf.space](https://ashutosh4902-product-image-classifier.hf.space)**

---

## 📚 References

- **E-commerce Product Images 18K** — Fatih Kögüş, Kaggle (2024). https://www.kaggle.com/datasets/fatihkgg/ecommerce-product-images-18k
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.
- TensorFlow / Keras Documentation — https://www.tensorflow.org/api_docs

---

## 👤 Author

**Ashutosh Patil**
MSc Business Intelligence & Data Science — University of Europe for Applied Sciences, Potsdam

📁 [GitHub Repository](https://github.com/Ashutoshpatil4902/SE01-PR-2026-Phase-II--III-Product-Image-classification-Ashutosh-Patil-.git) &nbsp;|&nbsp; 🏆 [Kaggle Notebook](https://www.kaggle.com/code/patilashutosh09/final-cnn-file) &nbsp;|&nbsp; 🌐 [Live Demo](https://ashutosh4902-product-image-classifier.hf.space)

---

*Machine Learning Course Project — June 2026*
