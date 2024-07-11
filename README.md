# Arabic Sign Language Image Classification

This project focuses on classifying Arabic sign language images using the YOLOv8 classification model. The dataset includes 32 classes representing different signs.

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
1. **Organizing the Dataset:** Images are organized into appropriate directories for training, validation, and testing.
2. **Splitting the Dataset:** The dataset is split into training, validation, and testing sets.

## Model Architecture
The model used for classification is the YOLOv8 classification model. The architecture includes:
- YOLOv8 backbone for feature extraction.
- Fully connected (dense) layers for classification.
- Softmax layer to output probabilities for each class.

## Training
- **Model:** YOLOv8 classification model (`yolov8n-cls.yaml` and `yolov8n-cls.pt`).
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- The model is trained on the training set with a validation split to monitor performance on unseen data.
- **Epochs:** 30
- **Image Size:** 64

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-cls.yaml")  # build a new model from YAML
model = YOLO("yolov8n-cls.pt")  # load a pretrained model
model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/kaggle/working/dataset", epochs=30, imgsz=64)
```
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.top1  # top1 accuracy
metrics.top5  # top5 accuracy

# Export the Model PyTorch Format ".pt"
model.export()

# Export the model ".onnx" format
model.export(format="onnx")

