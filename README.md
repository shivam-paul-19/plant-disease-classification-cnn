# 🌿 CropCare AI: Explainable Plant Disease Classification

CropCare AI is an end-to-end deep learning framework designed to diagnose crop leaf diseases with exceptional accuracy. By combining a **custom PyTorch Convolutional Neural Network (CNN)** with **Explainable AI (XAI) using Grad-CAM**, this project not only classifies plant diseases but also visualizes the exact pathological markers influencing the network's decisions. 

The project includes both a training pipeline (`plants.ipynb`) and a ready-to-deploy, premium interactive web dashboard (`app.py`) built with Streamlit.

---

## ✨ Key Features

- **High-Precision Custom CNN**: Built from scratch using PyTorch, achieving a validation accuracy and F1-score of **98.58%** on 17,572 validation samples.
- **Explainable AI (XAI) with Grad-CAM**: Real-time extraction of activation maps from the final convolutional block. It generates a dynamic heatmap overlay (Original vs. Heatmap comparison) showing exactly where the neural network is looking.
- **Premium User Dashboard**: Responsive, dark-themed user interface utilizing glassmorphic aesthetics, fluid layouts, custom CSS styling, and visual status badges.
- **Agronomic Advisory Engine**: A diagnostic summary coupled with an actionable, structured advisor providing **Symptoms**, **Organic Action Plans**, **Chemical Management Plans**, and **Preventative Guidelines** for all recognized conditions.
- **Scalable Support**: Out-of-the-box support for **14 crop types** and **38 distinct health classes** (both healthy leaves and complex infections).

---

## 📊 Dataset & Model Performance

### Dataset Information
The model was trained on the **New Plant Diseases Dataset (Augmented)**, a comprehensive expansion of the classic *PlantVillage* repository containing:
- **Total Images**: 87,867 crop leaf photos
- **Training Samples**: 70,295 images
- **Validation Samples**: 17,572 images
- **Classes**: 38 distinct crop-disease combinations across 14 plant varieties

### Model Validation Results
After training for 30 epochs using the Adam optimizer, the network achieved the following outstanding metrics:
- **Validation Accuracy**: `98.58%`
- **Weighted F1-Score**: `0.9858`
- **Macro F1-Score**: `0.9858`

*For detailed classification precision and recall scores per class, please refer to the end of the `plants.ipynb` notebook.*

---

## 🏗️ Neural Network Architecture

The custom PyTorch CNN uses a structured feature extraction architecture followed by a classification head:

1. **Input Layer**: accepts image tensors reshaped to $(224 \times 224 \times 3)$.
2. **Convolutional Feature Extractor**:
   - **Block 1**: Conv2D (16 channels, kernel=3, stride=2, padding=1) ➡️ ReLU ➡️ BatchNorm2D ➡️ MaxPool2D (kernel=2, stride=2)
   - **Block 2**: Conv2D (32 channels, kernel=3, stride=2, padding=1) ➡️ ReLU ➡️ BatchNorm2D ➡️ MaxPool2D (kernel=2, stride=1)
   - **Block 3**: Conv2D (64 channels, kernel=3, stride=1, padding=1) ➡️ ReLU ➡️ BatchNorm2D ➡️ MaxPool2D (kernel=2, stride=1)
   - **Block 4**: Conv2D (128 channels, kernel=3, stride=2, padding=2) ➡️ ReLU ➡️ BatchNorm2D ➡️ MaxPool2D (kernel=2, stride=1)
3. **Global Pooling**: Adaptive Average Pooling to output a $(1 \times 1)$ shape per channel.
4. **Classification Head**:
   - Flatten ➡️ Linear (128 ➡️ 64) ➡️ BatchNorm1d ➡️ ReLU ➡️ Dropout (rate=0.2) ➡️ Linear (64 ➡️ 38 classes)

---

## 📂 Directory Structure

```
plant-disease-classification-cnn/
├── app.py                     # Streamlit Explainable AI Dashboard
├── plants.ipynb               # PyTorch Model Training & Evaluation Notebook
├── plant_disease_model.pth    # Serialized weights of the trained CNN model
├── requirements.txt           # Python library dependencies
├── LICENSE                    # Project license terms
└── README.md                  # Comprehensive project documentation
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### 2. Install Dependencies
Install all required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Run the Web Dashboard
Start the premium Streamlit interactive app by executing:
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (usually `http://localhost:8501`) to start uploading crop images and diagnosing plant health.

---

## ⚙️ Training Setup (Reference)

If you wish to re-train the model, run the `plants.ipynb` notebook. The default hyperparameters are:
- **Optimizer**: `Adam` (Learning Rate = `0.0001`, Weight Decay = `1e-4`)
- **Loss Function**: `CrossEntropyLoss`
- **Batch Size**: `64`
- **Epochs**: `30`
- **Data Augmentation**: Random rotation (0, 90, 180, 270 degrees) and random horizontal flipping.

---

## 🛡️ Disclaimer
*CropCare AI is engineered for high-precision computer vision diagnostics. However, artificial intelligence classifications are subject to statistical margins of error. Please cross-reference predictions with regional agricultural extension standards or on-site professional agronomic evaluation before executing intensive or chemical crop treatments.*
