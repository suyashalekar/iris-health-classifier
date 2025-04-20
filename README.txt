# 🧠 Iris Image Health Classification (Multi-label Deep Learning)

This project builds a deep learning model that predicts health-related systems and symptoms from iris images using PyTorch.

## 🔍 Problem Statement
Each iris image can correspond to multiple health indicators (e.g., nervous system, pancreas weakness). This is a **multi-label classification task**, trained on a custom labeled dataset.

## 📁 Project Structure

- `iridology-my1.xlsx` → Excel file with image labels (main + sub categories)
- `images/` → Folder of iris images (named like `3.jpg`, `12.jpg`)
- `model_training.py` or `iris_pipeline.ipynb` → Complete data cleaning, training, and evaluation pipeline
- `multi_label_resnet18.pth` → Saved trained model
- `mlb.pkl` → MultiLabelBinarizer object for decoding labels
- `requirements.txt` → Python dependencies

## ⚙️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run training (if not already trained)
python model_training.py

# Step 3: Predict a sample image
python predict.py
