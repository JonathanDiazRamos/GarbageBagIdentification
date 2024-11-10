# Gothic: Image Classification and Transformation with CNN

Gothic is a machine learning project for binary image classification, specifically designed to distinguish between images with and without garbage bags. The repository includes custom data transformations and a convolutional neural network (CNN) model implemented in TensorFlow/Keras for image augmentation and classification.

## Features

- **Data Augmentation**: The code provides various transformations (e.g., horizontal flip, perspective distortion, Gaussian blur, rotation) to augment the dataset and enhance model generalization.
- **CNN Model for Binary Classification**: A CNN model is built using TensorFlow/Keras for binary classification with optimized hyperparameters.
- **Cyclical Learning Rate**: Utilizes a cyclical learning rate to improve model performance and avoid overfitting.

## Requirements

- Python 3.7+
- PyTorch
- TensorFlow
- TensorFlow Addons
- scikit-image
- NumPy
- Matplotlib

## Installation

To install the dependencies, run:
```bash
pip install torch tensorflow tensorflow-addons scikit-image numpy matplotlib
