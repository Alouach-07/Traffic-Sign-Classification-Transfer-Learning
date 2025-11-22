# 🚸 Traffic Sign Classification: Custom CNN vs. Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview

This project implements and compares different Deep Learning strategies for image classification using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The goal is to classify traffic signs into specific categories by evaluating the performance trade-offs between:

1.  **Training from Scratch:** A custom Convolutional Neural Network (CNN).
2.  **Transfer Learning:** Leveraging **MobileNetV2** pre-trained on ImageNet (Feature Extraction & Fine-Tuning).

This repository contains the source code, training logs, and links to the pre-trained models.

---

## 🔗 Resources & Downloads

Due to file size limits, the dataset and trained models are hosted externally.

| Resource | Link | Description |
| :--- | :--- | :--- |
| **Dataset** | [**Kaggle: GTSRB**](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) | Official dataset containing ~43 classes of traffic signs. |
| **Trained Models** | [**Google Drive Folder**](https://drive.google.com/drive/folders/155W8jqvsoxRnccQwKcASLOYyIsyCN7Jc?usp=sharing) | Contains `.keras` files for both the Baseline and Fine-Tuned models. |

---

## ⚙️ Methodology

### 1. Data Preprocessing

To focus the model learning and optimize resource usage, the dataset was filtered to retain **5 specific classes** (IDs: 1, 2, 12, 13, 14), representing:

* Speed limits (30km/h, 50km/h)
* Priority road
* Yield
* Stop

**Pipeline:**
* **Resizing:** Images scaled to `224x224` pixels (MobileNetV2 standard).
* **Augmentation:** Applied `Rotation`, `Width/Height Shift`, and `Zoom` to reduce overfitting.
* **Normalization:** Pixel values rescaled to `[0, 1]`.

### 2. Model Architectures

#### A. Baseline CNN (From Scratch)
A custom Sequential model featuring:
* 3 Convolutional blocks (Conv2D + ReLU + MaxPooling).
* Fully connected layers with Dropout (0.5) for regularization.
* Output layer with Softmax activation.

#### B. Transfer Learning (MobileNetV2)
* **Feature Extraction:** The base MobileNetV2 (trained on ImageNet) was frozen. A custom head (GlobalAveragePooling + Dense) was added.
* **Fine-Tuning:** The top **30 layers** of the base model were unfrozen and re-trained with a very low learning rate (`1e-5`) to adapt the pre-trained weights to the specific features of traffic signs.

---

## 📊 Performance Results

The models were evaluated on a validation set (20% split).

| Model Strategy | Training Epochs | Validation Accuracy | Validation Loss |
| :--- | :---: | :---: | :---: |
| **Custom CNN (Baseline)** | 15 | **99.26%** | **0.0297** |
| **MobileNetV2 (Frozen)** | 10 | 94.48% | 0.1411 |
| **MobileNetV2 (Fine-Tuned)** | 20 (Total) | **98.58%** | **0.0403** |

**Observation:** While the custom CNN performed exceptionally well on this restricted subset, MobileNetV2 with Fine-Tuning achieved comparable high accuracy and is likely more robust for generalizing to more complex real-world scenarios.

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.8+
* TensorFlow / Keras
* Pandas, NumPy, Matplotlib, Scikit-learn

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Alouach-07
/traffic-sign-classification.git](https://github.com/Alouach-07
/traffic-sign-classification.git)
    cd traffic-sign-classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Data & Models:**
    * Download the dataset from the [Kaggle Link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
    * (Optional) Download the `.keras` models from the [Google Drive Link](https://drive.google.com/drive/folders/155W8jqvsoxRnccQwKcASLOYyIsyCN7Jc?usp=sharing) and place them in the root directory.

4.  **Run the Notebook:**
    Open `Transfer Learning for Image Classification.ipynb` in Jupyter or Google Colab to view the training process and analysis.

---

## 📂 Project Structure

```text
.
├── Transfer Learning for Image Classification.ipynb  # Main analysis notebook
├── README.md                                         # Project documentation
├── .gitignore                                        # Git configuration
└── requirements.txt                                  # List of libraries
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/your-username/traffic-sign-classification/issues) if you want to contribute.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

## 👤 Author

**ALOUACH Abdennour**\
**Master Student** – Artificial Intelligence and Emerging Technologies  
*Context:* Advanced Machine Learning – Academic Project  
*Location:* Morocco
