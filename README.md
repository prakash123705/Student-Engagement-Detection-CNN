# Student-Engagement-Detection-Using-Facial-Expression-Recognition
Emotion-Driven Student Engagement Recognition System using CNN-based facial feature extraction and Random Forest classification to analyze classroom engagement from facial expressions.

# Emotion-Driven Student Engagement Detection System

## Overview

This project presents an **AI-powered Emotion-Driven Student Engagement Detection System** that analyzes classroom images to determine student engagement levels based on facial emotions.

The system combines:

* **Object Detection (YOLOv8)**
* **Face Detection (MTCNN & Haar Cascade)**
* **Feature Extraction (DenseNet121 / Custom CNN)**
* **Emotion Classification (Random Forest)**
* **Engagement Analysis (Rule-based mapping)**

---

## Key Features

Detects people and filters images containing faces
Extracts faces from classroom images
Generates **128-dimensional feature vectors**
Classifies emotions into 7 categories
Maps emotions → engagement levels
Calculates overall classroom engagement
Improves accuracy using **cosine similarity & hybrid metrics**
Handles misclassification refinement

---

## Emotion Classes

| Label | Emotion  |
| ----- | -------- |
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

---

## System Pipeline

### 1. Object Detection & Face Filtering

* YOLOv8 detects objects
* Faces detected inside "person" objects using Haar Cascade
* Only images containing faces are used for training

### 2. Face Extraction

* MTCNN detects faces in classroom images
* Faces are cropped and saved for processing

### 3. Feature Extraction

* DenseNet121 (ImageNet pretrained)
* Output converted to **128-dimensional feature vectors**

### 4. Model Training

* Random Forest Classifier
* Trained on extracted features dataset

### 5. Emotion Prediction

* Each face → predicted emotion

### 6. Engagement Mapping

* Emotions mapped to:

  * **Engaged** → Happy, Neutral, Surprise
  * **Disengaged** → Angry, Disgust, Fear, Sad

### 7. Classroom Engagement Calculation

* Majority voting + engagement ratio
* Final output:

  * Class Engaged / Disengaged
  * Engagement Level (High / Moderate / Low)

---

## Model Performance

* Model Accuracy: **~89%**

### Improvements Applied:

* Cosine similarity correction (Class 3 vs 4 confusion)
* Hybrid similarity:

  * Cosine
  * Euclidean
  * Mahalanobis

---

## Project Structure

```
├── data/
│   ├── dataset_images/
│   ├── extracted_faces/
│   └── DATASET__128Features_MultiLabels.csv
│
├── models/
│   ├── trained_rf_model.pkl
│   ├── emotion_rf.pkl
│   └── face_feature_extractor.keras
│
├── notebooks/
│   └── training_pipeline.ipynb
│
├── scripts/
│   ├── object_detection.py
│   ├── face_extraction.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── prediction.py
│   └── engagement_analysis.py
│
└── README.md
```

---

## Installation

```bash
pip install ultralytics opencv-python mtcnn tensorflow scikit-learn pandas numpy matplotlib joblib
```

---

## Usage

### Step 1: Detect faces in image

```python
# YOLO + Haar Cascade
```

### Step 2: Extract faces

```python
# MTCNN face detection
```

### Step 3: Extract features

```python
# DenseNet / CNN → 128 features
```

### Step 4: Train model

```python
# Random Forest
```

### Step 5: Predict emotions

```python
# Load model and predict
```

### Step 6: Calculate engagement

```python
result = calculate_class_engagement(predicted_emotions)
print(result)
```

---

## Engagement Logic

```python
Engaged = {Happy, Neutral, Surprise}
Disengaged = {Angry, Disgust, Fear, Sad}
```

### Engagement Levels

* **> 0.7** → Highly Engaged
* **0.4 – 0.7** → Moderately Engaged
* **< 0.4** → Low Engagement

---

## Misclassification Handling

Major confusion observed between:

* **Happy (3)** and **Sad (4)**

### Solution:

* Cosine similarity with class centroids
* Hybrid similarity approach:

  * Cosine
  * Euclidean
  * Mahalanobis

This improved accuracy significantly.

---

## Testing on New Images

* Extract features using CNN
* Load trained Random Forest model
* Predict emotion per face
* Aggregate for engagement analysis

---

## Future Improvements

* Use Deep Learning model (CNN/Transformer) instead of Random Forest
* Real-time video-based engagement detection
* Attention tracking + pose estimation
* Multi-modal analysis (audio + facial cues)
* Deploy as web app / dashboard

---

## Author

**[LOPINTI HARI]**
AI/ML Enthusiast | Data Science

