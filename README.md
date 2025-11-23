-----

````markdown
# Face Expression Detection using MobileNetV2

This project is a Deep Learning application capable of detecting and classifying human facial expressions into 7
 distinct categories. It utilizes
**Transfer Learning** with the **MobileNetV2** architecture to achieve high performance with a lightweight model,
 making it suitable for real-time applications.

## 📌 Introduction

Facial Emotion Recognition (FER) is a key technology in Human-Computer Interaction (HCI). This project moves beyond
simple CNNs by leveraging a pre-trained MobileNetV2 model. The system detects faces in an image/video, processes the
region of interest (ROI), and classifies the emotion using a fine-tuned neural network.

**Supported Emotions:**
1. Happy
2. Sad
3. Surprised
4. Angry
5. Fear
6. Neutral
7. Disgust

## ⚙️ The Pipeline

The system follows a strict processing pipeline from raw data to prediction:

```mermaid
graph TD;
    A[Input Image/Dataset] --> B[Face Detection];
    B -- Haarcascade --> C[Crop Face ROI];
    C --> D[Preprocessing];
    D -- Resize 96x96 --> E[RGB Conversion];
    E --> F[MobileNetV2 Feature Extraction];
    F --> G[Dense Layers & Dropout];
    G --> H[Softmax Classification];
    H --> I[Output Label];
````

### Process Breakdown:

1.  **Face Detection:** Uses OpenCV's Haarcascade (`haarcascade_frontalface_default.xml`) to isolate the face from the background.
2.  **Preprocessing:**
      * Resizes the face to **96x96** pixels.
      * Converts Grayscale images to **RGB** (3 channels) to satisfy MobileNet requirements.
      * Normalizes pixel values to the [0, 1] range.
3.  **Feature Extraction:** Passes the image through the **MobileNetV2** base (pre-trained on ImageNet).
4.  **Classification:** A custom Head (Dense layers) predicts the probability of each of the 7 emotions.

## 🧠 Model Architecture

We use **Transfer Learning** to adapt a powerful pre-existing model for our specific task.

  * **Base Model:** MobileNetV2 (Weights: ImageNet)
      * *Input Shape:* (96, 96, 3)
      * *State:* Top layers unfrozen for fine-tuning.
  * **Custom Head:**
      * `GlobalAveragePooling2D`: To reduce feature map dimensions.
      * `Dense (128 units, ReLU)`: For learning specific patterns.
      * `Dropout (0.5)`: To prevent overfitting.
      * `Dense (7 units, Softmax)`: Final output layer.

## 📂 Project Structure

```bash
FACE_EXPRESSION_DETECTION/
├── .venv/                   # Virtual Environment
├── unzipped/                # Dataset folder (Ignored by Git)
├── emotion_mobilenet_v2.h5  # The trained model file
├── requirements.txt         # Dependencies
├── unzip.py                 # Helper to extract dataset
├── Train_TransferLearning.py# Main training script (Fine-tuning)
├── test_for_moblieNet.py    # Inference script (Testing on images)
└── training_results.png     # Accuracy/Loss graphs
```

## 🚀 Getting Started

### 1\. Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd FACE_EXPRESSION_DETECTION
pip install -r requirements.txt
```

### 2\. Dataset Setup

Place your zipped dataset files inside the `unzipped/dataset` folder and run:

```bash
python unzip.py
```

### 3\. Training the Model

To train the model (or retrain from scratch):

```bash
python Train_TransferLearning.py
```

*This will generate the `emotion_mobilenet_v2.h5` file and a `training_results.png` graph.*

### 4\. Testing / Inference

To test the model on a specific image, update the `IMAGE_PATH` variable in the script and run:

```bash
python test_for_moblieNet.py
```

## 📊 Results

  * **Confusion Matrix:** Generated after training to visualize class accuracy.
  * **Training Strategy:** Fine-tuning with a low learning rate (`1e-5`) ensured the pre-trained weights were adapted to facial features without being destroyed.

## 🛠️ Tech Stack

  * **Language:** Python 3.12
  * **Deep Learning:** TensorFlow / Keras
  * **Computer Vision:** OpenCV
  * **Data Handling:** NumPy
  * **Visualization:** Matplotlib, Seaborn

<!-- end list -->

