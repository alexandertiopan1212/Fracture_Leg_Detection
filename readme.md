
# 🦴 Fracture Leg Detection using Deep Learning

**Automated Detection of Leg Fractures from Medical Images with VGG16-based CNN**

---

## 📌 Project Overview

Fractures in the lower extremities are common, and early, accurate diagnosis is critical for effective treatment. This project leverages **deep learning** and **transfer learning** with the **VGG16** convolutional neural network (CNN) to automatically detect **fractured vs. normal leg X-ray images**.

---

## 🗃️ Dataset

- **Total Images**: 142 images (128 for training, 14 for validation)
- **Classes**: 
  - `fracture`
  - `normal`
- **Structure**:
  ```
  dataset/
  ├── train/
      ├── fracture/
      └── normal/
  ```

---

## 🧠 Model Architecture

The model is built on **VGG16** as the base (with frozen weights), extended by:

- Additional `Conv2D`, `MaxPooling2D`, and `GlobalAveragePooling2D` layers
- Fully connected `Dense` layers with **BatchNormalization** and **Dropout**
- Final output layer: 2 neurons (sigmoid) for **binary classification**

### 🔧 Compilation:

- **Optimizer**: SGD (Nesterov, learning rate 0.01)
- **Loss**: Binary Crossentropy
- **Metrics**:
  - Accuracy, Precision, Recall, AUC
  - True/False Positives/Negatives

---

## 🔄 Data Augmentation

Used `ImageDataGenerator` with:

- Rotation, shearing, zoom, flipping (H/V)
- Brightness adjustment
- Preprocessing: `preprocess_input` from VGG16

This increases data diversity and robustness.

---

## 📊 Model Training

- **Epochs**: 500 (with early stopping)
- **Batch Size**: 32
- **Callbacks**:
  - `ModelCheckpoint`: save best model
  - `EarlyStopping`: monitor training stagnation
- **Train/Validation Split**: 90% / 10%

The model was trained and evaluated on small batches with performance monitored across epochs.

---

## 📈 Training Visualization

Accuracy and loss were visualized using `matplotlib`, showing learning progression. While there were fluctuations (possibly due to small dataset size), general improvement in performance was observed.

---

## 🧪 Testing & Prediction

The trained model is used to predict leg fracture images with the following logic:

```python
img = load_img("path_to_image", target_size=(256, 256))
arr = img_to_array(img)
arr = preprocess_input(arr)
prediction = model.predict(np.array([arr]))
```

Predicted class is mapped to either:
- `fracture` (label 0)
- `normal` (label 1)

The output is visualized with true and predicted labels for validation.

---

## ✅ Results & Key Takeaways

- The model shows potential for binary classification of leg fractures.
- With more annotated data and regularization techniques, performance can be improved.
- This approach can be useful in **telemedicine**, **emergency triage**, and **rural diagnostics**.

---

## 🛠 Tech Stack

- Python
- TensorFlow & Keras
- VGG16 Pre-trained Model
- Matplotlib & ImageDataGenerator

