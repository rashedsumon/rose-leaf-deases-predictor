import streamlit as st
import os
import torch
from data_loader import get_classes_and_counts
from model import get_model, predict_image

# App configurations
st.set_page_config(page_title="Rose Leaf Disease Analyzer", page_icon="🌹", layout="centered")

st.title("🌹 Rose Leaf Disease Classifier")
st.write("Upload a clear photo of a rose leaf to analyze its health status using AI.")

# --- STEP 1: Manage Dataset and Class Resolution ---
@st.cache_resource
def setup_application_assets():
    """Cache data and architecture structure to accelerate app reloads"""
    data_dir, classes = get_classes_and_counts()
    return classes

try:
    class_labels = setup_application_assets()
except Exception as e:
    class_labels = ["Healthy Leaf Rose", "Rose Rust", "Rose Sawfly or Rose Slug"]

# --- STEP 2: Load Model State ---
@st.cache_resource
def load_trained_weights(num_classes):
    model = get_model(num_classes=num_classes)
    
    # If you have pre-saved trained weights file locally, uncomment lines below:
    # if os.path.exists("rose_model.pth"):
    #     model.load_state_dict(torch.load("rose_model.pth", map_location=torch.device('cpu')))
    
    return model

model = load_trained_weights(len(class_labels))

# --- STEP 3: Graphical Interface Layout ---
uploaded_file = st.file_uploader("Choose a leaf image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Rose Leaf Workspace", use_column_width=True)
    
    st.write("")
    run_analysis = st.button("🔍 Run Diagnostics AI")
    
    if run_analysis:
        with st.spinner("Analyzing structural cellular patterns..."):
            try:
                label, confidence = predict_image(uploaded_file, model, class_labels)
                
                # Highlight outcome based on health status
                if "healthy" in label.lower():
                    st.success(f"**Result:** {label} ")
                    st.balloons()
                else:
                    st.error(f"**Condition Detected:** {label} ")
                    st.warning("⚠️ Recommendation: Isolate plant leaf stems and treat using recommended biological agents.")
            except Exception as e:
                st.exception(f"Error handling image parsing pipeline: {e}")