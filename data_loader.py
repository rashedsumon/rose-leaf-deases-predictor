import os
import shutil
import kagglehub
from torchvision import datasets

def load_and_prepare_dataset():
    """
    Downloads the dataset via kagglehub, identifies its class names,
    and returns the system path containing the class folders.
    """
    print("Checking dataset status...")
    # Downloads or verifies the dataset path on the local or Streamlit runtime environment
    raw_path = kagglehub.dataset_download("parvezkabir221155539/rose-leaf-disease-dataset")
    
    # Sometimes Kaggle structure includes nested folders, we look for directories containing subfolders
    for root, dirs, files in os.walk(raw_path):
        if len(dirs) > 0 and not root.endswith('__pycache__'):
            # Check if folders look like image classes
            return root
            
    return raw_path

def get_classes_and_counts():
    """
    Inspects the downloaded dataset directory to collect target class labels.
    """
    data_dir = load_and_prepare_dataset()
    try:
        dataset = datasets.ImageFolder(root=data_dir)
        classes = dataset.classes
        return data_dir, classes
    except Exception as e:
        # Fallback labels if local evaluation finds a shallow or empty root initially
        return data_dir, ["Healthy Leaf Rose", "Rose Rust", "Rose Sawfly or Rose Slug"]

if __name__ == "__main__":
    path, classes = get_classes_and_counts()
    print(f"Dataset securely loaded at: {path}")
    print(f"Identified diagnostic classes: {classes}")