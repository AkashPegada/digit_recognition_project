# 🧠 Digit Recognition using CNN & Grad-CAM

An interactive deep learning web application built using **Streamlit** that recognizes handwritten digits with a **Convolutional Neural Network (CNN)** and visually explains its predictions using **Grad-CAM**.

---

## 🎯 Project Overview
This project was developed as part of **CS5720 – Neural Networks and Deep Learning** coursework.  
It demonstrates a complete deep learning workflow from model training to explainability.

Users can:
- 🖼️ **Upload** or ✏️ **Draw digits**
- 🔢 View **Top-3 Predictions** with confidence scores
- 🔥 See **Grad-CAM heatmaps** to understand model focus
- ⚙️ Run both a **Streamlit web app** and an optional **Tkinter desktop GUI**

---

## ⚙️ Tech Stack
| Component | Technology |
|------------|-------------|
| Framework | TensorFlow / Keras |
| Frontend | Streamlit |
| Visualization | Matplotlib, Grad-CAM |
| Utilities | OpenCV (headless), Pillow, NumPy |
| Language | Python 3.10 + |

---

## 🧩 Folder Structure
```text
digit_recognition_project/
│── streamlit_app.py          # Main Streamlit app
│── digit_predictor.py        # Model loader & top-3 prediction logic
│── gradcam_visualizer.py     # Grad-CAM implementation
│── digit_gui.py              # Optional Tkinter GUI
│── digit_recognizer.h5       # Pre-trained CNN model
│── requirements.txt          # Dependencies
│── .gitignore
└── README.md
```

🚀 How to Run
1️⃣ Clone the repository
```bash
git clone https://github.com/AkashPegada/digit_recognition_project.git
cd digit_recognition_project
```

2️⃣ Create & activate a virtual environment

Windows (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
4️⃣ Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501
 in your browser.

🧠 CNN Model Summary

| Layer Type          | Parameters |
| ------------------- | ---------- |
| Conv2D (32 filters) | ReLU       |
| Conv2D (64 filters) | ReLU       |
| MaxPooling2D        | 2×2        |
| Dropout             | 0.25       |
| Flatten             | —          |
| Dense (128)         | ReLU       |
| Dropout             | 0.5        |
| Dense (10)          | Softmax    |

Accuracy: ~99% on MNIST test data.

🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights areas the CNN focused on when making predictions.

How to use:

Enable “Show Grad-CAM heatmap” in the sidebar.

If it errors, click “Show model layers” → copy the last Conv2D layer name (e.g., conv2d_2) and paste it into “Last conv layer (optional)”.

Observe the heatmap overlay that visualizes model attention 🔥.

🧪 Optional — Retrain the Model

If digit_recognizer.h5 is missing or you want to retrain, run the following script:
```python
# train_mnist_model.py
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train[..., None]/255.0, x_test[..., None]/255.0

model = models.Sequential([
    layers.Conv2D(32,3,activation='relu',input_shape=(28,28,1)),
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_split=0.1)
model.save("digit_recognizer.h5")
```

📦 Requirements

streamlit>=1.37
tensorflow>=2.20,<2.21
opencv-python-headless
numpy
pillow
matplotlib
streamlit-drawable-canvas
h5py

🧰 .gitignore

.venv/
venv/
venv310/
__pycache__/
.ipynb_checkpoints/
.DS_Store
*.png

📸 Screenshots
| Upload Digit                                | Grad-CAM Heatmap                               |
| ------------------------------------------- | ---------------------------------------------- |
| ![digit_sample.png](digit_recognition_project/digit_sample.png) | ![gradcam\_example](digit_recognition_project/gradcam_sample.png) |



🏁 Author

👤 Akash Pegada
🎓 CS5720 – Neural Networks & Deep Learning
📍 University of Central Missouri

© 2025 Akash Pegada — All Rights Reserved.
