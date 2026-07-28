# Bayesian Uncertainty-Aware Audio Content Classification using Wav2Vec2

> **Ongoing Individual Research**

A research-oriented framework for environmental sound classification that extends **Wav2Vec2** with **Bayesian Monte Carlo Dropout** and **calibration analysis** to estimate prediction uncertainty and improve the reliability of deep learning models.

---

## Motivation

Modern deep learning models often produce highly confident predictions, even when they are incorrect. While classification accuracy is important, real-world AI systems must also communicate **how confident they are** in their predictions.

This project investigates uncertainty estimation and calibration techniques for self-supervised audio transformers, enabling models to provide reliable confidence estimates alongside class predictions.

---

## Objectives

- Develop a robust environmental sound classification framework using self-supervised audio representations.
- Quantify predictive uncertainty using Bayesian inference.
- Evaluate model confidence through calibration analysis.
- Build an uncertainty-aware inference pipeline suitable for research and real-world deployment.

---

## Features

- Fine-tuned **Wav2Vec2** for environmental sound classification
- ESC-50 dataset support
- Bayesian Monte Carlo Dropout inference
- Predictive confidence estimation
- Predictive entropy computation
- Predictive variance estimation
- Expected Calibration Error (ECE)
- Reliability Diagram generation
- Confusion Matrix visualization
- Classification Report generation
- Modular training, evaluation, and inference pipeline

---

## Project Pipeline

```
                Audio Input
                     │
                     ▼
        Audio Preprocessing (Librosa)
                     │
                     ▼
     Self-Supervised Wav2Vec2 Encoder
                     │
                     ▼
         Audio Classification Head
                     │
                     ▼
        Bayesian Monte Carlo Dropout
                     │
                     ▼
      Multiple Stochastic Predictions
                     │
                     ▼
        Mean Prediction Distribution
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
   Confidence     Entropy      Variance
                     │
                     ▼
          Calibration Analysis
          (ECE & Reliability Diagram)
```

---

## Dataset

**ESC-50**

- 50 environmental sound classes
- 2,000 labeled audio clips
- Standard benchmark for environmental sound classification

---

## Results

| Metric | Score |
|---------|------:|
| Accuracy | **75.25%** |
| Precision | **78.19%** |
| Recall | **75.25%** |
| F1 Score | **75.31%** |
| Expected Calibration Error (ECE) | **0.1192** |

---

## Repository Structure

```
audio-content-classification/
│
├── models/
│   ├── audio_classifier.py
│   └── classifier.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── calibration.py
│   └── reliability.py
│
├── utils/
│   └── uncertainty.py
│
├── predict.py
├── requirements.txt
└── README.md
```

---

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Wav2Vec2
- Librosa
- Scikit-learn
- NumPy
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Mahek3108/audio-content-classification.git
cd audio-content-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python -m training.evaluate
```

The evaluation pipeline reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Expected Calibration Error (ECE)
- Classification Report
- Confusion Matrix

---

## Inference

```bash
python predict.py --audio path/to/audio.wav
```

Example output:

```
Prediction : Cat
Confidence : 97.75%
Predictive Entropy : 0.1405
Predictive Variance : 0.000029
```

---



Ongoing Individual Research


## Project Status

This repository represents the current public milestone of an ongoing individual research project.

The code is shared for academic demonstration and portfolio purposes. Future research developments and experimental extensions are being developed separately.

---

## Author

**Mahek Shaikh**

AI Engineer

Research Interests:
- Deep Learning
- Audio AI
- Bayesian Deep Learning
- Trustworthy AI
- Uncertainty Quantification
