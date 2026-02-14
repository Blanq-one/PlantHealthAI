# PlantHealthAI

PlantHealthAI is a comprehensive system designed for the detection and classification of plant diseases. By leveraging Convolutional Neural Networks (CNNs), transfer learning with MobileNet, and TensorFlow Lite, the project provides a lightweight and efficient solution for real-time agricultural monitoring.

## Features

- Custom CNN-based classification for targeted disease identification
- Transfer learning using MobileNet for enhanced feature extraction and accuracy
- TensorFlow Lite integration for optimized edge deployment
- Raspberry Pi camera support for on-site image acquisition

## System Components

### Convolutional Neural Network (cnn_model.py)
The custom CNN architecture is designed for binary disease classification. It utilizes a sequence of convolutional and pooling layers followed by fully connected layers, optimized using the Adam objective function and binary cross-entropy loss.

### Transfer Learning Module (disease_model.py)
This component employs a pre-trained MobileNet model as a feature extractor. The network is augmented with custom dense layers and dropout for robust multi-class disease classification, utilizing categorical cross-entropy for training.

### TensorFlow Lite Inference (tflite_inference.py)
For deployment on resource-constrained devices, the models are converted to TensorFlow Lite format. This script handles model loading, image preprocessing via the PIL library, and efficient inference execution.

### Real-Time Capture (pi_camera.py)
A dedicated script for the Raspberry Pi environment that interfaces with the PiCamera module. It captures high-resolution images and saves them locally for immediate processing by the inference engine.

## Directory Structure

| File | Description |
| :--- | :--- |
| cnn_model.py | Custom CNN training and architecture definition |
| disease_model.py | MobileNet transfer learning implementation |
| tflite_inference.py | Optimized inference script for TFLite models |
| pi_camera.py | Raspberry Pi camera integration utility |

## Technical Requirements

The project requires Python 3.x and the following dependencies:

- tensorflow
- keras
- numpy
- pillow
- tflite-runtime

Installation can be performed using:
`pip install tensorflow keras numpy pillow tflite-runtime`
