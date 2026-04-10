
# Breath Sound Classification

This project focuses on classifying respiratory sounds into inhale and exhale using deep learning.  
The goal is to build a robust audio classification model by combining feature extraction techniques with a hybrid neural network architecture.

---

## Overview
This project aims to classify human breathing sounds into two categories: inhale and exhale.  
Respiratory sounds are an important physiological signal and can be used in medical applications such as detecting respiratory disorders.

The project explores the full deep learning pipeline, including audio preprocessing, feature extraction, and model design.

---

## Dataset
- Audio dataset in WAV format
- Training samples: 4,000 (balanced between inhale and exhale)
- Test samples: 1,000 (unlabeled)
- Labels:  
  - Inhale → 'I'  
  - Exhale → 'E'

---

## Method

### Audio Preprocessing
- Resampled all audio to 16kHz
- Fixed audio length to 3 seconds
- Applied zero-padding for shorter samples

### Feature Extraction
- Converted audio signals into Mel-Spectrogram
- Used 64 Mel frequency bins
- Applied log scaling for normalization

### Model Architecture
The model is a hybrid deep learning structure combining CNN and GRU:

- CNN Feature Extractor:
  - Conv2D → BatchNorm → ReLU → Pooling
  - Channel progression: 1 → 32 → 64 → 128
  - Adaptive pooling to stabilize feature size

- Bi-directional GRU:
  - Captures temporal patterns in audio sequences
  - Hidden size: 128
  - Bidirectional structure

- Fully Connected Layer:
  - Linear → ReLU → Dropout → Linear
  - Final binary classification output

---

## Training Strategy
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (learning rate = 0.001)
- Batch size: 32
- Epochs: 20

Training accuracy improved rapidly:
- ~50% at initial stage
- ~80% within 5 epochs
- ~99% at final stage

---

## Results
- Final training accuracy: ~99%
- Loss stabilized below 0.05

The model successfully learned meaningful patterns from audio signals, although further validation is required to ensure generalization.

---

## Experimental Insights
- Mel-Spectrogram effectively captures time-frequency characteristics
- Combining CNN and GRU improves both spatial and temporal feature learning
- Fast convergence observed during training

---

## Limitations
- No validation split → potential overfitting
- Limited evaluation on unseen data

---

## Future Work
- Introduce validation set and cross-validation
- Apply data augmentation (noise injection, time shift)
- Explore advanced architectures (LSTM, Transformer)
- Use pretrained audio models (e.g., wav2vec 2.0)
- Apply ensemble strategies

---

## Tech Stack
- Python
- PyTorch
- Librosa
- NumPy
- Pandas

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
