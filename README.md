# MNIST Digit Classification — Deep Learning (MLP, CNN, SVM)

## Overview
Built and compared three machine learning approaches for handwritten digit classification on the MNIST dataset — a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN), and a Support Vector Machine (SVM) baseline. The goal was to understand the trade-offs in architecture complexity and classification performance on a standardised 70,000-image benchmark.

## Dataset
- **Source:** MNIST (Modified National Institute of Standards and Technology)
- **Training set:** 60,000 images | **Test set:** 10,000 images
- **Image size:** 28×28 pixels, grayscale
- **Classes:** 10 (digits 0–9)

## Architecture — CNN (best performing model)

```
Input (28×28×1)
  → Conv2D(32, 3×3, ReLU) → BatchNormalization → MaxPooling(2×2)
  → Conv2D(64, 3×3, ReLU) → BatchNormalization → MaxPooling(2×2)
  → Conv2D(128, 3×3, ReLU) → BatchNormalization
  → Flatten
  → Dense(128, ReLU) → Dropout(0.5)
  → Dense(10, Softmax)
```

**Optimizer:** Adam | **Loss:** Sparse Categorical Crossentropy
**Epochs:** 15 | **Batch size:** 64 | **Validation split:** 10%

## Results

| Model | Test Accuracy |
|-------|--------------|
| MLP (baseline) | ~97.8% |
| CNN (this model) | ~99.2% |
| SVM | ~98.6% |

The CNN outperformed both baselines by leveraging spatial feature extraction through convolutional layers. BatchNormalization accelerated training stability and Dropout(0.5) reduced overfitting.

## Key techniques
- **Convolutional feature extraction** — spatial pattern learning across 3 Conv2D layers
- **Batch Normalisation** — stabilised training and improved convergence speed
- **Dropout regularisation** — prevented overfitting on training data
- **SVM comparison** — established a strong non-deep-learning baseline

## Tech stack
`Python` `TensorFlow` `Keras` `scikit-learn` `NumPy`
