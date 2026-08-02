# MNIST Handwritten Digit Recognition

## Project Overview

This project recognizes handwritten digits (0–9) using the MNIST dataset. A Convolutional Neural Network (CNN) was developed to classify handwritten digit images with high accuracy. The project also includes a Tkinter GUI that allows users to upload an image of a handwritten digit and receive the predicted digit with a confidence score.

## Features

- Image preprocessing
- Data normalization
- Data augmentation
- Early stopping
- CNN model
- Model evaluation
- Confusion Matrix
- Classification Report
- Accuracy and Loss graphs
- Tkinter GUI

## Dataset

Dataset: MNIST Handwritten Digits

- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 classes (0–9)

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Tkinter

## Workflow

1. Load the MNIST dataset
2. Normalize pixel values
3. Reshape images
4. Apply data augmentation
5. Build a CNN model
6. Train using Early Stopping
7. Evaluate model performance
8. Save the trained model
9. Predict handwritten digits using the GUI

## CNN Architecture

The model consists of:

- Convolution Layers
- Max Pooling Layers
- Dropout Layers
- Dense Layers
- Softmax Output Layer

## Model Evaluation

Evaluation metrics include:

- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Confusion Matrix
- Classification Report

## GUI

The GUI allows users to:

- Upload a handwritten digit image
- Automatically preprocess the image
- Predict the digit
- Display confidence score

## Future Improvements

- Support recognition of multiple digits
- Develop a web application
- Train with additional handwritten datasets
- Improve preprocessing for noisy images

## Author

Nancy Batra
