import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os

class CFG:
    model_name = 'resnet18'
    target_size = 6
    size = 256

class CustomNet(nn.Module):
    def __init__(self, model_name=CFG.model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.classifier = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x

def get_transforms():
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomNet(pretrained=False)
    
    if not os.path.exists(model_path):
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])    
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def predict(image_path, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_transforms()
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_preds = model(image_tensor)
        
    prediction = y_preds.softmax(1).argmax(1).item()
    return prediction
