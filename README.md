# Shoplifting Video Detection Using Deep Learning

This project implements a **video-based shoplifting detection system** using deep learning.  
It aims to automatically classify short surveillance clips as **Shoplifting** or **Non-Shoplifting** behavior by analyzing temporal motion and spatial visual cues.

---

## Project Overview

Shoplifting detection in real-time video streams is a challenging task that requires understanding both **spatial** and **temporal** features.  
To tackle this, the project explores several deep learning architectures:

- **2D CNN + LSTM**
- **2D CNN + GRU** ✅ *(Best performing model)*
- **3D CNN**

Although **Object Detection** and **Human Pose Estimation (HPE)** were initially explored, these approaches were set aside due to performance limitations and model complexity.  
The final pipeline focuses on **sequence-based action recognition** using **2D CNN + GRU** for efficient and accurate classification.

---

## Dataset

- The dataset is organized into two main classes:
```
Shop DataSet/
├── non shop lifters/
└── shop lifters/
```
- Each folder contains surveillance video clips (`.mp4`, `.avi`, `.mov`).
- Clips are preprocessed to have a fixed number of frames (default: 16).

---

## Preprocessing

Each video undergoes the following steps before being passed to the model:

1. **Frame Extraction** → Read all frames from each video.  
2. **Frame Sampling** → Uniformly sample 16 frames to represent the entire clip.  
3. **Transformations & Augmentation**
 - Resize to **256×256**
 - Random horizontal flips, brightness & contrast jittering, and rotation (for training only)
 - Normalize pixel values to `[-1, 1]`
4. **Tensor Stacking** → Frames are stacked into tensors of shape `[T, C, H, W]`.

---

## Model Architectures

### **2D CNN + GRU (Final Model)**

Each frame is processed by a CNN for feature extraction, and the sequence of frame features is passed to a GRU to capture temporal motion.
```
Input Video → CNN → Flatten → GRU → FC → Sigmoid → Binary Classification
```
**Architecture Summary**
| Component | Description |
|------------|-------------|
| CNN | 2 Conv layers (32→64 filters) + ReLU + MaxPooling + AdaptiveAvgPool(4×4) |
| GRU | Input: 1024, Hidden: 256, Bidirectional |
| FC Layer | Linear(512 → 1) |
| Activation | Sigmoid (binary classification) |

---

## Training Setup

- **Loss Function:** Binary Cross-Entropy with Logits  
- **Optimizer:** Adam  
- **Learning Rate:** Started at `1e-3`, later fine-tuned to `1e-4`  
- **Batch Size:** 4  
- **Epochs:** 20 total (trained in 4 runs × 5 epochs)  
- **Train/Val Split:** 80% / 20%

Each run progressively improved accuracy and reduced validation loss.

---

## 📊 Results

| Metric | Validation |
|:-------|:-----------:|
| **Accuracy** | 99.17% |
| **Precision** | 98.51% |
| **Recall** | 100% |
| **F1 Score** | 99.25% |

**Confusion Matrix**
| | Predicted Non-Shoplifting | Predicted Shoplifting |
|:--|:--:|:--:|
| **Actual Non-Shoplifting** | 54 | 1 |
| **Actual Shoplifting** | 0 | 66 |

✅ The **2D CNN + GRU** model achieved the highest performance with minimal overfitting.

---

## Training Curves

Loss and accuracy curves across multiple training runs:

- Continuous improvement with learning rate tuning.
- Stable convergence without overfitting.
<img width="1389" height="590" alt="download" src="https://github.com/user-attachments/assets/992e5d68-63fd-479d-9833-0c8ba8b66eb4" />

---

## Evaluation Metrics

The evaluation includes:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

All metrics computed using **scikit-learn**.

---

## Environment & Dependencies

| Library | Version |
|----------|----------|
| Python | 3.10+ |
| PyTorch | 2.x |
| torchvision | 0.16+ |
| OpenCV | 4.x |
| NumPy | 1.26+ |
| Matplotlib | 3.x |
| scikit-learn | 1.x |
| tqdm | — |

---

## Future Work

- Integrate **Object Detection (YOLO/SSD)** for region-based analysis.  
- Revisit **Human Pose Estimation (HPE)** for behavioral pattern recognition.  
- Deploy the trained model with **Flask + HTML/CSS** for real-time inference.  
