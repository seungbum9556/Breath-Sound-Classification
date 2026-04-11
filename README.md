# Breath Sound Classification

A deep learning model for classifying respiratory sounds (inhale vs exhale) using Mel-Spectrogram features and a CNN + Bi-GRU architecture.

---

## Overview
This project aims to classify human breathing sounds into two categories: inhale and exhale.  
Respiratory sounds are an important physiological signal and can be used in medical applications such as detecting respiratory disorders.

---

## Dataset
- Audio dataset in WAV format
- Training samples: 4,000 (balanced)
- Test samples: 1,000 (unlabeled)
- Labels: Inhale (I), Exhale (E)

---

## Method

### Audio Preprocessing
- Resampled audio to 16kHz
- Fixed length to 3 seconds
- Applied zero-padding

### Feature Extraction
- Mel-Spectrogram (64 Mel bins)
- Log scaling for normalization

### Model Architecture
- CNN feature extractor (Conv2D → BatchNorm → ReLU → Pooling)
- Bi-directional GRU for temporal modeling
- Fully connected layers for classification

---

## Training Strategy
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 20

---

## Results
- Training Accuracy: ~99% (rapid convergence)
- Loss: < 0.05
- Note: No validation split → potential overfitting

---

## Experimental Insights
- Mel-Spectrogram effectively captures time-frequency patterns
- CNN + GRU improves spatial and temporal feature learning
- Fast convergence observed

---

## Key Contributions
- Built an audio preprocessing pipeline (resampling, padding, Mel-Spectrogram)
- Designed a CNN + Bi-GRU hybrid model
- Conducted experiments and analyzed model behavior

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
