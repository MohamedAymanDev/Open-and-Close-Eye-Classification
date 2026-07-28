# 👁️ Eye State Classification AI

## 🧠 AI-Powered Eye Open/Closed Detection System

Eye State Classification is a deep learning-based computer vision application that detects whether an eye is **open or closed** using a Convolutional Neural Network (CNN).

The system allows users to upload an eye image and get an AI prediction with confidence score and probability visualization through an interactive Streamlit interface.

---

## 🚀 Project Features

✅ Detects eye state (Open / Closed)  
✅ CNN-based image classification  
✅ Image preprocessing using OpenCV  
✅ Confidence score prediction  
✅ Probability visualization  
✅ Interactive Streamlit deployment  
✅ Modern UI with animations and responsive design  

---

## 🧠 Deep Learning Model

The model is built using a custom Convolutional Neural Network (CNN).

### Model Architecture:

- Convolutional Layers
- Max Pooling Layers
- Flatten Layer
- Fully Connected Dense Layers
- Sigmoid Output Layer

### Input Shape:

```
224 × 224 × 1
```

### Classes:

```
0 → Closed Eye
1 → Open Eye
```

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras
- CNN

### Computer Vision
- OpenCV
- Pillow

### Data Processing
- NumPy
- Pandas
- Scikit-learn

### Deployment
- Streamlit

---

## 📂 Project Structure

```
Eye-State-Classification/

│
├── app.py
├── label_encoder.pkl
├── requirements.txt
├── README.md
│
└── assets/
    ├── home.png
    └── prediction.png
```

---

## ⚙️ How To Run

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/Eye-State-Classification.git
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run Application

```bash
streamlit run app.py
```

---

## 🖼️ Application Preview

(Add screenshots here)

Example:

```
assets/home.png
assets/prediction.png
```

---

## 📊 Prediction Output

The application provides:

- Predicted eye state
- Confidence percentage
- Probability visualization

Example:

```
Prediction:
Open Eye

Confidence:
98.7%
```

---

## 🎯 Project Goal

The goal of this project is to build an AI system capable of automatically identifying eye state from images using deep learning techniques.

Potential applications:

- Driver drowsiness detection
- Eye tracking systems
- Human-computer interaction
- Computer vision applications

---

## 👨‍💻 Author

**Mohamed Ayman**

Aspiring Machine Learning Engineer | Artificial Intelligence Student
