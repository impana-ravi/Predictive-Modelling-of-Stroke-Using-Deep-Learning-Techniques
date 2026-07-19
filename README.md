# 🧠 Predictive Modelling of Stroke Using Deep Learning Techniques

A Deep Learning-based web application that predicts brain stroke types from MRI images using **Convolutional Neural Networks (CNN)** models. The system classifies MRI scans into **Haemorrhagic Stroke, Ischemic Stroke, and Normal** categories using transfer learning approaches with **VGG16 and ResNet50**.

The application provides a user-friendly Flask web interface where users can upload MRI images and receive predictions based on trained deep learning models.

---

# 📌 Project Overview

Stroke is a critical medical condition where early detection plays an important role in reducing complications and improving treatment outcomes.

This project presents a Deep Learning-based stroke detection system that analyzes brain MRI images and classifies them into three categories:

- 🩸 Haemorrhagic Stroke
- 🧠 Ischemic Stroke
- ✅ Normal (No Stroke)

The system uses transfer learning with **VGG16** and **Resnet50** models for accurate MRI image classification.
A Flask-based web application is developed to provide real-time prediction, report generation, and personalized health recommendations through email.

---

# 🎯 Problem Statement

Traditional stroke diagnosis mainly depends on manual interpretation of MRI scans by radiologists, which can be time-consuming and may involve human errors.

There is a need for an automated system that can:
- Analyze MRI images efficiently
- Classify different stroke types accurately
- Provide faster preliminary diagnosis support
- Generate useful health guidance for users

---

# 🚀 Objectives

- Develop Deep Learning Models for Stroke Classification.
- Perform Comparative Analysis of VGG16 and ResNet50. 
- Design a User-Friendly Web Application. 
- Build a web-based interface for easy image upload and prediction. Implement Automated Report Generation with Health Recommendations.

---

# ✨ Features

### 🔐 User Authentication
- User registration
- User login using email

### 🧠 MRI Image Prediction
- Upload brain MRI image
- Image preprocessing
- Deep learning-based prediction

### 📊 Stroke Classification

The model predicts:

- Haemorrhagic Stroke
- Ischemic Stroke
- Normal Brain

### 📄 Report Generation

The system generates a detailed report containing:

- MRI image
- Predicted stroke category
- Health recommendations

### 📧 Email Support

The generated report and recommendations are sent to the user's registered email address.

### 🌐 User-Friendly Web Application

- Simple and responsive interface
- Easy image upload
- Fast prediction results

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Deep Learning Frameworks
- TensorFlow
- Keras
- Convolutional Neural Networks (CNN)
- Transfer Learning

## Models
- VGG16
- ResNet50

## Web Framework
- Flask

## Frontend
- HTML
- CSS
- JavaScript

## Database
- MySQL
  
## Development Tools

- VS Code
- Python 3.7

---

# 🏗️ System Workflow

```text
User Registration/Login
          ↓
Upload MRI Image
          ↓
Image Preprocessing
          ↓
Deep Learning Model
(VGG16 / ResNet50)
          ↓
Stroke Classification
          ↓
Generate Report
          ↓
Send Report and health recommendations via Email Report
```

---

# 🤖 Deep Learning Models

## VGG16

VGG16 is a deep convolutional neural network architecture used for image classification. Transfer learning is applied using pretrained features to classify MRI images into stroke categories.

## ResNet50

ResNet50 uses residual learning connections that help train deeper networks and improve feature extraction for medical image classification.

---

# 📊 Model Performance Analysis

The models were evaluated using:

- Accuracy
- Loss
- Precision
- Recall
- F1-score
- Confusion Matrix

A comparative analysis was performed between **VGG16** and **ResNet50** to identify the better-performing architecture for stroke classification.

| Model | Analysis |
|------|----------|
| VGG16 | Accuracy, loss and classification performance evaluated |
| ResNet50 | Accuracy, loss and classification performance evaluated |

The final model selection was based on accuracy, convergence and generalization performance.

---

# 📂 Project Structure

```text
Predictive-Modelling-of-Stroke-Using-Deep-Learning-Techniques/

│
├── static/
│   ├── bg.png
│   ├── bg2.jpeg
│   ├── jquery.min.js
│   ├── main.js
│   ├── style.css
│   └── table.css
│
├── templates/
│   ├── index.html
│   ├── uploadimage.html
│   ├── userlogin.html
│   └── userreg.html
│
├── app.py
├── resnetandvgg.py
├── transfer_learning_vgg_16.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📚 Dataset

The project uses brain MRI images categorized into:

## Classes

1. Haemorrhagic Stroke
2. Ischemic Stroke
3. Normal

The dataset is used for training and testing deep learning models.

**Note:** The MRI dataset is not included in this repository due to its large size.

---

# ⚙️ Installation and Execution

## Clone Repository

```bash
git clone https://github.com/impana-ravi/Predictive-Modelling-of-Stroke-Using-Deep-Learning-Techniques.git
```

## Navigate to Project Folder

```bash
cd Predictive-Modelling-of-Stroke-Using-Deep-Learning-Techniques
```

## Install Required Dependencies

```bash
pip install -r requirements.txt
```

## Add Trained Models

The trained model files are excluded due to large file size.

Add locally:

```bash
best_model.h5
best_model2.h5
```

before running the application.

## Run Application

```bash
python app.py
```

---

# 📈 Outcomes

- Accurate classification of MRI scans into Haemorrhagic, Ischemic, and Normal categories.
- VGG16 vs ResNet50 performance comparison to find the most effective model.
- Flask-based web app for real-time MRI upload and stroke prediction.
- Automated report with MRI image, result, and personalized recovery tips.
- Fast, low-cost decision-support tool for doctors and researchers.
- Promotes AI integration in healthcare for early diagnosis and better patient care.

---

# 🔮 Future Enhancements

- Deploy the application on cloud platforms.
- Improve accuracy using advanced architectures.
- Add Explainable AI techniques like Grad-CAM.
- Expand dataset with more MRI samples.
- Integrate with hospital healthcare systems.

---

# 👩‍💻 Authors

**Impana R**  
**Harshitha D V**  
**Monisha S**

🎓 Bachelor of Engineering  
Department of Computer Science and Engineering  
BGS Institute of Technology

GitHub:
https://github.com/impana-ravi

LinkedIn:
https://www.linkedin.com/in/impanaravi

Email:
impanaravi2005@gmail.com

---

# 🙏 Acknowledgement

This project was developed as an **academic project** to explore the application of Deep Learning and Computer Vision techniques in healthcare for automated stroke detection from MRI images. It demonstrates the use of AI-based medical image classification for assisting early stroke detection.
