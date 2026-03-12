# Helmet Detection

A deep learning project that classifies whether a person in an image is wearing a helmet or not, using Convolutional Neural Networks (CNNs) built with PyTorch.

## Overview

This project uses a [Kaggle helmet detection dataset](https://www.kaggle.com/andrewmvd/helmet-detection) containing 764 labeled images across two categories:

- **With Helmet** (label: 0)
- **Without Helmet** (label: 1)

The primary motivation is road and construction site safety — this classifier can be integrated into automated safety monitoring systems.

## Project Structure

```
helmet-detection/
├── data/                        # Preprocessed cropped images and CSVs
│   ├── objects.csv              # Bounding box info, cropped image names, labels
│   └── images.csv               # Original image names
├── preparation.ipynb            # Data preprocessing and EDA
├── final-project.ipynb          # Model training and evaluation
├── final-report.md              # Detailed project report
├── project_proposal.md          # Initial project proposal
├── progress_report.md           # Mid-project progress update
└── 23f_hw04.pdf                 # Assignment reference
```

## Methodology

### Data Preprocessing
The raw dataset consists of 764 images with XML annotation files containing bounding box coordinates and labels. Images were cropped to their bounding boxes, resized to **45×45 pixels**, and stored alongside two CSV files (`objects.csv`, `images.csv`). An 80/20 train-validation split was applied based on original images to prevent data leakage.

### Model Architecture

A custom `ObjectClassifier` class was used to systematically experiment with CNN architectures, varying:
- Number of convolutional layers
- Kernel sizes and output channels
- Dropout rates
- Loss functions (`NLLLoss` vs `CrossEntropyLoss`)

All models shared:
- **Activation**: ReLU (conv layers), LogSoftmax (output)
- **Pooling**: Max pooling with kernel size 2
- **FC layers**: 128 → 2 units
- **Optimizer**: Adam (lr = 0.001)
- **Epochs**: 50 | **Batch size**: 32

### Model Comparison

| Model | Conv Layers | Loss Function | ROC-AUC | Test Accuracy |
|-------|-------------|---------------|---------|---------------|
| Model 1 | 2 (16, 32 ch) | NLLLoss | **0.812** | 0.884 |
| Model 2 | 3 (16, 32, 64 ch) | NLLLoss | 0.628 | 0.894 |
| Model 3 | 3 (16, 32, 64 ch) | CrossEntropyLoss | 0.787 | 0.893 |
| Model 4 | 2 (16, 32 ch) | CrossEntropyLoss | 0.803 | 0.880 |

> ROC-AUC was prioritized over accuracy to minimize false positives (i.e., classifying someone without a helmet as wearing one).

## Results

**Model 1** achieved the best ROC-AUC score of **0.812** with a test accuracy of **88.4%**. The model performs well overall but struggles with edge cases where a person is wearing some other head covering that resembles a helmet — a limitation tied to the small dataset size.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- matplotlib
- Pillow

Install dependencies:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow
```

## Usage

1. **Data Preparation**: Run `preparation.ipynb` to preprocess raw images and generate the `data/` directory with cropped images and CSVs.
2. **Training & Evaluation**: Run `final-project.ipynb` to train the CNN models and evaluate performance.

## Dataset

The dataset is sourced from Kaggle: [Helmet Detection by Andrew MVD](https://www.kaggle.com/andrewmvd/helmet-detection). Download and place the raw images in the appropriate directory before running the preprocessing notebook.

## Author

**Harshvardhan Bhatnagar** — [GitHub](https://github.com/harshvardhanb25)
