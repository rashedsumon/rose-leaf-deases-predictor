import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def get_model(num_classes=3):
    """
    Initializes a lightweight MobileNetV3 model with a custom final classifier head.
    Perfect for edge deployment and cloud environments.
    """
    # Using modern weights initialization standard for torchvision
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    
    # Replace the classification head to match our dataset class count
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

def get_transforms():
    """
    Defines image transformations required by MobileNetV3 backbones.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def predict_image(image_path_or_buf, model, classes):
    """
    Preprocesses an incoming user-uploaded image and computes the AI model's diagnosis.
    """
    model.eval()
    transform = get_transforms()
    
    # Load and format image safely
    img = Image.open(image_path_or_buf).convert("RGB")
    img_t = transform(img).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, dim=0)
        
    return classes[index.item()], confidence.item() * 100