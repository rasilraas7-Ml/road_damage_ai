# 🚧 RoadGuard AI — Road Damage Detection

RoadGuard AI is a computer vision project for detecting potholes and road damage from images, videos, and live webcam input.

The project compares two deep learning approaches:

- PyTorch Faster R-CNN
- TensorFlow MobileNetV2 Grid Detector

The main web application uses the PyTorch Faster R-CNN detector.

---

## 🚀 Features

- Pothole detection from images
- Bounding-box visualization
- Confidence score for each detection
- Video road-damage detection
- Live webcam detection
- PyTorch object detection pipeline
- TensorFlow detection experiment
- Model evaluation
- Confidence threshold analysis
- PyTorch vs TensorFlow comparison
- Flask web application
- OpenCV image/video processing

---

## 🧠 Models

### PyTorch

Faster R-CNN with ResNet-50 FPN backbone.

Used as the primary detector for the Flask application.

### TensorFlow

MobileNetV2-based grid detector.

Used as a lightweight comparison model.

---

## 📊 Dataset

The project uses a pothole detection dataset containing YOLO-format annotations.

Dataset statistics:

- Training images: 1,581
- Validation images: 396
- Total images: 1,977
- Total pothole annotations: 7,337
- Classes: 1
- Class: Pothole

The dataset itself is not included in this repository.

---

## 🛠️ Tech Stack

- Python
- PyTorch
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Flask
- Matplotlib
- Pillow
- Scikit-learn

---

## 📁 Project Structure

```text
road-damage-ai/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── data/
│
├── models/
│   ├── pytorch/
│   └── tensorflow/
│
├── outputs/
│
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── pytorch/
│   ├── tensorflow/
│   ├── evaluation/
│   └── utils/
│
├── requirements.txt
├── .gitignore
└── README.md