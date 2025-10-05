# Video Classification for Stolen Content Detection

A deep learning project implementing three architectures from scratch to classify videos as original or stolen content.

## Project Overview

This project provides a complete pipeline for video-based binary classification using three different neural network architectures:
- **3D CNN**: Direct spatial-temporal convolutions on video volumes
- **CNN-RNN**: Frame-wise CNN feature extraction + LSTM temporal modeling
- **Transformer**: CNN backbone + self-attention mechanism for temporal dependencies

## Dataset

**Location**: `/content/data/Shop Dataset` or `/content/Shop_DataSet.zip`

**Structure**:
```
Shop Dataset/
├── original/     # Original videos (label 0)
│   ├── video1.mp4
│   └── ...
└── stolen/       # Stolen videos (label 1)
    ├── video1.mp4
    └── ...
```

## Features

### Data Processing
- Automatic video extraction and frame sampling (16 frames per video)
- Uniform temporal sampling across video duration
- Frame resizing to 112×112 pixels
- Comprehensive data augmentation:
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jittering (brightness, contrast, saturation, hue)
  - Random resized crop
  - Gaussian blur

### Model Architectures

#### 1. 3D CNN
- 6 convolutional blocks with batch normalization
- Progressive channel expansion: 3 → 64 → 128 → 256 → 512
- 3D max pooling for spatial-temporal down sampling
- ~30M parameters

#### 2. CNN-RNN
- 2D CNN encoder for per-frame feature extraction
- Bidirectional LSTM for temporal sequence modeling
- 2-layer LSTM with hidden size 256
- ~25M parameters

#### 3. Video Transformer
- CNN backbone for spatial features
- 6-layer transformer encoder with 8 attention heads
- Learnable positional encoding
- CLS token for classification
- ~35M parameters

## Installation

```bash
# Required packages
pip install torch torch vision OpenCV-python NumPy matplotlib scikit-learn seaborn tqdm pandas
```

## Usage

### Quick Start

```python
# Run complete pipeline
python video_classification.py
```

### Step-by-Step

```python
# 1. Analyze dataset
from data Analyzer import extract_and_analyze
analyzer, df = extract_and_analyze()

# 2. Train models
from video classification import main
main()
```

### Custom Configuration

```python
# Edit in main() function:
DATA_PATH = '/content/data/Shop DataSet'
NUM_FRAMES = 16          # Frames per video
IMG_SIZE = (112, 112)    # Frame dimensions
BATCH_SIZE = 4           # Adjust for GPU memory
NUM_EPOCHS = 30          # Training epochs
LEARNING_RATE = 0.001    # Initial learning rate
```

## Training Details

- **Loss Function**: Cross-entropy
- **Optimizer**: Adam with weight decay (1e-4)
- **Learning Rate Schedule**: Step decay (×0.5 every 10 epochs)
- **Gradient Clipping**: Max norm 1.0
- **Data Split**: 64% train / 16% validation / 20% test

## Output Files

```
/content/
├── augmentation_examples.png          # Augmentation visualization
├── training_comparison.png            # Training curves
├── dataset_analysis.png               # Dataset statistics
├── best_3d_cnn.pth                   # Best model weights
├── best_cnn-rnn.pth                  # Best model weights
├── best_transformer.pth              # Best model weights
├── 3d_cnn_confusion_matrix.png       # Test results
├── cnn-rnn_confusion_matrix.png      # Test results
├── transformer_confusion_matrix.png   # Test results
├── preprocessing_config.json          # Dataset configuration
└── video_info.csv                     # Video metadata
```

## Performance Expectations

| Model | Parameters | Training Time/Epoch | Expected Accuracy |
|-------|-----------|---------------------|-------------------|
| 3D CNN | ~30M | 2-3 min (GPU) | 85-90% |
| CNN-RNN | ~25M | 1-2 min (GPU) | 87-92% |
| Transformer | ~35M | 2-3 min (GPU) | 90-95% |

*Results vary based on dataset quality and size*

## Troubleshooting

### Out of Memory Errors
```python
BATCH_SIZE = 2        # Reduce batch size
NUM_FRAMES = 8        # Reduce frames
IMG_SIZE = (64, 64)   # Reduce resolution
```

### Slow Training
```python
NUM_FRAMES = 8        # Fewer frames
NUM_EPOCHS = 15       # Fewer epochs
```

### Poor Accuracy
- Increase training epochs (50-100)
- Add more augmentation
- Verify label quality
- Check for data imbalance
- Tune hyperparameters

## Model Inference

```python
import torch
from video_classification import VideoTransformer, VideoDataset
from torch.utils.data import DataLoader

# Load trained model
model = VideoTransformer(num_classes=2)
model.load_state_dict(torch.load('/content/best_transformer.pth'))
model.eval()

# Prepare test video
test_dataset = VideoDataset(['test_video.mp4'], [0], num_frames=16, augment=False)
test_loader = DataLoader(test_dataset, batch_size=1)

# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    for video, _ in test_loader:
        video = video.to(device)
        output = model(video)
        prob = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        
        label = 'Stolen' if predicted.item() == 1 else 'Original'
        confidence = prob[0][predicted.item()].item() * 100
        
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
```

## Project Structure

```
video-classification/
├── video_classification.py    # Main training script
├── data_analyzer.py           # Dataset analysis tools
├── README.md                  # This file
└── requirements.txt           # Dependencies
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (recommended for GPU training)
- 8GB+ GPU memory (for batch size 4)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{video_classification_2025,
  title={Video Classification for Stolen Content Detection},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/video-classification}}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- 3D CNN architecture inspired by C3D and I3D networks
- Transformer implementation based on "Attention is All You Need"
- Video preprocessing techniques from TSN and TSM papers


## Future Improvements

- [ ] Add temporal segment networks (TSN)
- [ ] Implement video swin transformer
- [ ] Add mixed precision training
- [ ] Support for longer videos (>30 seconds)
- [ ] Model ensemble methods
- [ ] Real-time inference optimization
- [ ] Multi-class classification support
- [ ] Transfer learning from pretrained models

## Version History

- **v1.0.0** (2025): Initial release with three architectures
  - 3D CNN implementation
  - CNN-RNN hybrid model
  - Transformer-based model
  - Complete data pipeline

  - Comprehensive evaluation tools
