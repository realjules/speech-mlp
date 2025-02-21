# Frame-Level Speech Recognition

This project implements a frame-level speech recognition system using a Multi-Layer Perceptron (MLP) neural network. The model processes MFCC (Mel-frequency cepstral coefficients) features to predict phonemes at each time frame.

## Project Structure

```
mlp/
├── requirements.txt
├── train.ipynb              # Training notebook
└── src/
    └── speech_recognition/
        ├── __init__.py
        ├── data/           # Data loading and preprocessing
        │   ├── __init__.py
        │   ├── dataset.py
        │   └── augment.py
        ├── models/         # Neural network models
        │   ├── __init__.py
        │   └── mlp.py
        └── utils/          # Utility functions
            ├── __init__.py
            └── training.py
```

## Features

- Frame-level phoneme recognition using MLP
- MFCC feature processing (28 features per frame)
- Data augmentation techniques:
  - Random time shifting (-25ms to +25ms)
  - Frequency masking
  - Time masking
- Training optimizations:
  - Adan optimizer
  - CosineAnnealing and ReduceLROnPlateau schedulers
  - Progressive reduction of augmentation and dropout

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Open `train.ipynb` in Jupyter Notebook/Lab
2. Follow the training process step by step
3. The notebook includes:
   - Data loading and preprocessing
   - Model configuration
   - Training loop
   - Evaluation metrics

## Model Performance

The model achieves:
- Training accuracy: 74.87%
- Validation accuracy: 87.537%

## Training Strategy

The training process involves multiple stages:

1. Initial training with Adan optimizer (lr=1e-3) and CosineAnnealing
2. Fine-tuning with ReduceLROnPlateau scheduler
3. Adjustment of time shift range
4. Progressive reduction of dropout and augmentation