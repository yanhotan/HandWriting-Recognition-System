# Handwriting Recognition System

This project implements a handwriting recognition system that integrates **image preprocessing using fuzzy logic**, **genetic algorithms for feature selection**, and **transfer learning models (InceptionV3, ResNet50)** for classification. The system identifies handwriting owners from cropped and preprocessed handwriting images.

---

## **Table of Contents**
1. [Features](#features)
2. [Setup](#setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results and Visualizations](#results-and-visualizations)
7. [Acknowledgments](#acknowledgments)

---

## **Features**
- **Image Preprocessing with Fuzzy Logic**:
  - Enhances images using brightness, sharpness, edge density, texture complexity, and other features.
- **Genetic Algorithm for Feature Selection**:
  - Optimizes feature selection based on Logistic Regression and Random Forest classifiers.
- **Hybrid Neural Network**:
  - Combines image features with transfer learning models (InceptionV3, ResNet50).
- **Dynamic Learning Rate Adjustment**:
  - Uses fuzzy logic to adjust learning rates during training.
- **Interpretability Tools**:
  - Generates GradCAM, saliency maps, and classification reports.

---

## **Setup**
### **Requirements**
- Python 3.7+
- TensorFlow 2.x
- scikit-fuzzy
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- DEAP (Genetic Algorithm)

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Additional Setup**
- Clone the dataset repository:
  ```bash
  git clone https://github.com/Walmond3/WIX3001-Alt-Ass
  ```

---

## **Dataset Preparation**
1. **Dataset Structure**:
   - Images should be stored in a directory named `Handwriting` inside the cloned repository.

2. **Image Cropping**:
   - The script crops images into smaller pieces and applies fuzzy logic preprocessing.

3. **Feature Extraction**:
   - Features such as HOG, LBP, Hu Moments, and statistical measures (mean, variance, skewness, etc.) are extracted for analysis and classification.

---

## **Implementation Details**

### **1. Image Preprocessing**
- **Fuzzy Logic 1**:
  - Adjusts contrast based on brightness and sharpness.
- **Fuzzy Logic 2**:
  - Enhances contrast using additional parameters like edge density, texture complexity, saturation, and hue.

### **2. Genetic Algorithm (GA) for Feature Selection**
- **Fitness Functions**:
  - Logistic Regression and Random Forest classifiers.
- **Selection, Mutation, and Crossover**:
  - Implements tournament/rank selection, bit-flip/gaussian mutation, and two-point crossover.
- **Stopping Criteria**:
  - Achieves fitness > 0.998 or no improvement over 40 generations.

### **3. Hybrid Model**
- Combines:
  - **Transfer Learning Models** (InceptionV3, ResNet50): Trained on cropped handwriting images.
  - **Extracted Features**: Fused with the neural network for enhanced classification.

### **4. Dynamic Learning Rate**
- Adjusts learning rate dynamically using a fuzzy control system based on error rates during training.

---

## **Usage**

1. **Run Image Cropping and Preprocessing**
   ```python
   python preprocess_images.py
   ```

2. **Feature Selection Using Genetic Algorithm**
   ```python
   python feature_selection_ga.py
   ```

3. **Train the Model**
   - Example: ResNet50
   ```python
   python train_model.py --model resnet50
   ```
   - Example: InceptionV3
   ```python
   python train_model.py --model inceptionv3
   ```

4. **Evaluate the Model**
   - Generates classification reports, confusion matrices, and visualization tools like GradCAM and saliency maps.

---

## **Results and Visualizations**

### **Performance Metrics**
- **Owner Accuracy**:
  - Per-owner classification accuracy.
- **Alphanumeric Accuracy**:
  - Character recognition performance for digits and letters.

### **Visualizations**
- **Loss and Accuracy Trends**:
  - Training and testing performance across epochs.
- **GradCAM and Saliency Maps**:
  - Highlight regions of interest for interpretability.
- **Confusion Matrix**:
  - Detailed breakdown of model predictions.

---

## **Acknowledgments**
This project was developed as part of an academic assignment for **WIX3001 - AI and Machine Learning Applications**. Special thanks to all contributors and resources utilized in building this comprehensive handwriting recognition system.
