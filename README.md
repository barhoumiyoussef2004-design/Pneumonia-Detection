# 🫁 Pneumonia Detection from Chest X-Rays

> Binary image classification using MobileNetV2 transfer learning — trained on the Kaggle Chest X-Ray Images (Pneumonia) dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Keras](https://img.shields.io/badge/Keras-Transfer%20Learning-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-92.86%25-brightgreen?style=flat-square)

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **92.86%** |
| Test Loss | 0.2804 |
| Fine-tune Train Accuracy | ~94.57% |
| Fine-tune Val Accuracy | ~83.05% |

---

## Approach

Two-phase transfer learning on MobileNetV2 pretrained on ImageNet:

**Phase 1 — Feature Extraction**
Freeze all MobileNetV2 layers. Train only the classification head on top. Lets the model adapt to the domain without destroying pretrained weights.

**Phase 2 — Fine-Tuning**
Unfreeze the full base model. Continue training with a low learning rate (`1e-5`) and `ReduceLROnPlateau` to avoid overshooting. 10 epochs, batch size 32.

The key preprocessing fix: the dataset contains grayscale images, which were converted to 3-channel RGB before being fed into MobileNetV2 (which expects `(224, 224, 3)` input). Skipping this conversion causes a shape mismatch crash at runtime.

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Paul Mooney, Kaggle.

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
└── test/
```

~5,800 X-ray images split across train/val/test. Binary labels: `NORMAL` vs `PNEUMONIA`.

---

## Project Structure

```
pneumonia-detection/
├── notebooks/
│   └── pneumonia_detection.ipynb   # Full pipeline: EDA → training → evaluation
├── src/
│   ├── data.py                     # Data loading, augmentation, RGB conversion
│   ├── model.py                    # MobileNetV2 model definition
│   └── train.py                    # Training loop, callbacks
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt
```

Download the dataset from Kaggle and place it under `data/chest_xray/`, or update the path in the notebook.

---

## Requirements

```
tensorflow>=2.10
numpy
matplotlib
scikit-learn
kaggle
```

---

## Key Implementation Notes

- **Grayscale → RGB**: chest X-rays are single-channel; converted via `tf.image.grayscale_to_rgb` before passing to MobileNetV2
- **Data augmentation**: horizontal flip, zoom, slight rotation applied during training to combat overfitting
- **Callbacks**: `ReduceLROnPlateau` (patience=2) triggered twice during fine-tuning, reducing LR from `1e-5` → `3e-6` → `1e-6`
- **Class imbalance**: training set is skewed toward PNEUMONIA (~3:1); handled via class weights

---

## License

MIT
