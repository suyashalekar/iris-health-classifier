import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import pickle

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 67)  # total number of categories
    model.load_state_dict(torch.load("multi_label_resnet18.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Load MultiLabelBinarizer from pickle ---
@st.cache_data
def load_label_binarizer():
    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)
    return mlb

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Prediction Function ---
def predict(image, model, mlb, threshold=0.5):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output).squeeze().numpy()
    predicted_indices = [i for i, p in enumerate(probs) if p > threshold]
    predicted_labels = [mlb.classes_[i] for i in predicted_indices]
    return predicted_labels

# --- Streamlit UI ---
st.title("🔍 Iris Image Category Predictor")
st.write("Upload an iris image and get predicted health-related categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    mlb = load_label_binarizer()

    st.write("Predicting...")
    labels = predict(image, model, mlb)

    st.success("**Predicted Labels:**")
    for label in labels:
        st.write(f"• {label}")
