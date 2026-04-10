# 🩺 Diabetic Retinopathy Detection using CNN

A deep learning project that uses a Convolutional Neural Network (CNN) to automatically detect and classify **Diabetic Retinopathy (DR)** from retinal fundus images. This project is based on the research paper *"Convolutional Neural Network for Diabetic Retinopathy Detection"* presented at **ICAIS 2021 (IEEE)**.

---

## 📌 Problem Statement

Diabetic Retinopathy is a leading cause of blindness caused by damage to blood vessels in the retinal tissue of diabetic patients. Early detection is critical, but manual diagnosis requires expert ophthalmologists. This project automates DR detection using deep learning, enabling faster and more accessible screening.

---

## 📂 Project Structure

```
Diabetic_retinopathy_CNN/
│
├── Diabetes/
│   └── Dataset/
│       ├── Mild/           # Retinal images with Mild DR
│       └── Moderate/       # Retinal images with Moderate DR
│
└── BASE PAPER.pdf          # Reference IEEE research paper (ICAIS 2021)
```

---

## 🧠 Model Overview

The model uses a **Convolutional Neural Network (CNN)** architecture trained to perform binary classification — detecting whether a patient has Diabetic Retinopathy or not.

### Architecture
The CNN consists of three main types of layers:
- **Convolutional Layers** — Feature extraction from retinal images
- **Pooling Layers** — Dimensionality reduction
- **Fully Connected Layers** — Classification output

### Optimizer
- **RMSprop**

---

## 📊 Dataset

- **Source:** [APTOS 2019 Blindness Detection Dataset (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection/data) + IDRiD dataset images
- **Training samples:** 3,789 retinal images
- **Testing samples:** 948 retinal images (80/20 split)
- **Classes:**
  - `Class 0` — Normal (No DR)
  - `Class 1` — Diabetic Retinopathy

### Local Dataset (included in this repo)
| Class    | No. of Images |
|----------|---------------|
| Mild     | 20            |
| Moderate | 24            |

---

## ⚙️ Data Preprocessing

1. **Image Resizing** — Standardizing input image dimensions
2. **Pixel Rescaling** — Pixel values normalized to [0, 1] by dividing by 255 (improves Sigmoid activation performance)
3. **Label Encoding** — Labels converted to machine-readable numeric format

---

## 🔁 Data Augmentation

To improve generalization and handle class imbalance, the following augmentation techniques were applied:

- Cropping
- Shifting
- Rotation
- Padding
- Horizontal & Vertical Flipping
- Zoom

---

## 📈 Results

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 96.15%  |
| Sensitivity   | 79%     |
| Precision     | 89%     |
| F1-Score      | 84.1%   |
| AUC (ROC)     | 0.82    |
| Loss          | 0.0839  |

### Confusion Matrix (on test set)

|                  | Predicted Normal | Predicted DR |
|------------------|------------------|--------------|
| **Actual Normal**| 310              | 55           |
| **Actual DR**    | 120              | 463          |

### Comparison with Existing Work

| Model              | Training Data | Optimizer | Accuracy |
|--------------------|---------------|-----------|----------|
| Kele Xu et al.     | 1,000 images  | Adam      | 94.5%    |
| **Proposed Model** | **4,737 images** | **RMSprop** | **96.15%** |

---

## 🛠️ Requirements

```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python pillow
```

> Python 3.7+ recommended

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Diabetic_retinopathy_CNN.git
   cd Diabetic_retinopathy_CNN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   - Place retinal images under `Diabetes/Dataset/` in appropriate class folders (`Mild`, `Moderate`, etc.)

4. **Train the model**
   ```bash
   python train.py
   ```

5. **Evaluate / Predict**
   ```bash
   python predict.py --image path/to/retinal_image.jpg
   ```

---

## 📚 Reference

> Shital N. Firke, Ranjan Bala Jain, *"Convolutional Neural Network for Diabetic Retinopathy Detection"*, Proceedings of ICAIS 2021, IEEE. DOI: [10.1109/ICAIS50930.2021.9395796](https://doi.org/10.1109/ICAIS50930.2021.9395796)

---

## 📝 License

This project is for academic and educational purposes. Dataset usage is subject to the [APTOS Kaggle competition terms](https://www.kaggle.com/c/aptos2019-blindness-detection/rules).

---

## 🙌 Acknowledgements

- IEEE ICAIS 2021 for the base research paper
- Kaggle APTOS 2019 Blindness Detection Challenge for the dataset
- IDRiD (Indian Diabetic Retinopathy Image Dataset) for supplementary images
