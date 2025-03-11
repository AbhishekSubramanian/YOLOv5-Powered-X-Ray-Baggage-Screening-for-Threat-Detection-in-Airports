# YOLOv5-Powered-X-Ray-Baggage-Screening-for-Threat-Detection-in-Airports

## Project Summary
Ensuring accurate classification of luggage is crucial for security in public spaces and transportation hubs. This project implements a **CV-Driven Automated Suspicious Baggage Detection** system using the **YOLOv5** object detection model. The model is trained to detect and classify suspicious objects within luggage scans, leveraging YOLOv5's robust feature extraction capabilities.

The project evaluates the model's performance using precision, recall, and mAP metrics on a diverse dataset. The findings demonstrate the system's reliability and potential for integration into automated security protocols, reducing reliance on manual inspections by Transportation Security Officers (TSOs).

## Features
- **Computer Vision-based Object Detection**: Utilizes YOLOv5 for real-time detection.
- **Pretrained Model Fine-tuning**: Transfer learning on X-ray images of luggage.
- **Performance Optimization**: Adjustments to hyperparameters to enhance precision and recall.
- **Deployment**: Integrated into a **Streamlit** web application for real-time detection.

---

## Installation and Setup
Follow these steps to run the notebook and use the detection system.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/CVASBD.git
cd CVASBD
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv cvasbd_env
source cvasbd_env/bin/activate  # On Windows use `cvasbd_env\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Ensure that the required packages such as **torch, torchvision, numpy, OpenCV, pandas, and Streamlit** are installed.

### 4. Download Pretrained YOLOv5 Model
Before running the detection, download the YOLOv5 model weights:
```bash
wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -P models/
```

Alternatively, you can train your own model using the dataset.

### 5. Running the Jupyter Notebook
Launch Jupyter Notebook and open the **CVASBD.ipynb** file:
```bash
jupyter notebook
```
Execute all the cells to run the object detection pipeline.

### 6. Running the Streamlit Web Application
For real-time baggage detection, use the **Streamlit** app:
```bash
streamlit run app.py
```
This will launch a local web application where you can upload X-ray images and receive detection results.

---

## Dataset
The dataset used is based on the **OPIXray Dataset**, consisting of annotated X-ray images of luggage containing objects such as:
- Scissors
- Folding Knives
- Straight Knives
- Utility Knives
- Multitool Knives

Preprocessing includes **image resizing (640×640), augmentation (flipping, scaling, HSV adjustments), and anchor box optimization** to enhance detection accuracy.

---

## Model Training and Evaluation
- **Baseline Model:** YOLOv5s trained for 40 epochs with initial hyperparameters.
- **Fine-Tuned Model:** Trained for 95 epochs with optimized learning rate, classification loss weight, and IoU threshold.
- **Performance Metrics:**
  - **Baseline:** Precision = 0.867, Recall = 0.859, mAP@0.5 = 0.873
  - **Fine-Tuned:** Precision = 0.903, Recall = 0.871, mAP@0.5 = 0.894

---

## Challenges and Improvements
- **Data Imbalance:** Addressed through **augmentation and class-specific tuning**.
- **Training Instability:** Resolved NaN losses by validating dataset integrity.
- **Model Performance:** Improved detection accuracy for underrepresented classes.

---
## Acknowledgments
- YOLOv5 by Ultralytics
- OPIXray Dataset contributors

