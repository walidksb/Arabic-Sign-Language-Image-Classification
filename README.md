# Arabic Sign Language Image Classification

This project focuses on classifying Arabic sign language images using a neural network. The dataset includes 32 classes representing different signs.

## Table of Contents
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Dataset
The dataset consists of images divided into training and testing sets. The directory structure is as follows:

train/train/haik-image classification/ # Training images
test/test/ # Testing images

## Data Preprocessing
1. **Loading Images:** Images are loaded from the specified directories.
2. **Resizing Images:** All images are resized to a consistent shape to feed into the neural network.
3. **Normalization:** Pixel values are normalized to the range [0, 1].
4. **Data Augmentation:** Data augmentation techniques such as rotation, flipping, and zooming are applied to increase the diversity of the training data.

## Model Architecture
The model used for classification is a Convolutional Neural Network (CNN). The architecture includes:
- Multiple convolutional layers with ReLU activation.
- Max-pooling layers to reduce the spatial dimensions.
- Fully connected (dense) layers for classification.
- Softmax layer to output probabilities for each class.

## Training
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- The model is trained on the training set with a validation split to monitor performance on unseen data.

## Evaluation
The trained model is evaluated on the test set to measure its performance. Key metrics include accuracy, precision, recall, and F1-score.

## Results
The model achieves high accuracy in classifying the Arabic sign language images. Detailed results and performance metrics are provided in the notebook.

## Conclusion
The project successfully demonstrates the classification of Arabic sign language using a neural network. Future work can explore more sophisticated models and larger datasets to improve performance further.

