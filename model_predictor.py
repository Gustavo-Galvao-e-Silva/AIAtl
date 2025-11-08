"""
Model predictor module for loading and using the trained recyclable classifier model.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


def load_model(model_path, num_classes=None, device=None):
    """
    Load the trained MobileNet v3 Small model.
    
    Args:
        model_path: Path to the saved model .pth file
        num_classes: Number of classes (if None, will try to infer from data directory)
        device: Device to load model on (CPU/GPU)
    
    Returns:
        Loaded model, device, class names
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get class names from training data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "splitData", "train")
    
    if os.path.exists(train_path):
        from torchvision.datasets import ImageFolder
        temp_dataset = ImageFolder(train_path, transform=transforms.ToTensor())
        class_names = temp_dataset.classes
        num_classes = len(class_names)
    else:
        # Fallback: assume standard recyclable categories
        class_names = ['cardboard', 'e-waste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile', 'trash']
        if num_classes is None:
            num_classes = len(class_names)
    
    # Load model architecture with ImageNet weights first (same as training)
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    
    # Freeze backbone (same as training)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
    
    # Load trained weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.to(device)
    model.eval()
    
    return model, device, class_names


def get_transform():
    """
    Get the same transform used for validation/test (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_image(model, image, class_names, device, transform=None):
    """
    Predict the recyclable category for a single image.
    
    Args:
        model: Trained model
        image: PIL Image or image path
        class_names: List of class names
        device: Device to run inference on
        transform: Image transform (if None, uses default)
    
    Returns:
        Dictionary with prediction, confidence, and top predictions
    """
    if transform is None:
        transform = get_transform()
    
    # Load image if it's a path
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        # Assume it's already a PIL Image
        image = image.convert('RGB')
    
    # Preprocess image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
    
    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
    
    results = {
        'predicted_class': class_names[predicted_idx],
        'confidence': float(confidence),
        'predicted_material': class_names[predicted_idx],  # For compatibility with website
        'product_type': class_names[predicted_idx],  # For compatibility with website
        'top_predictions': [
            {
                'class': class_names[idx.item()],
                'confidence': float(prob.item())
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
    }
    
    return results


# Global model cache
_model_cache = None
_device_cache = None
_class_names_cache = None


def get_model(script_dir=None):
    """
    Get or load the model (with caching for efficiency in Streamlit).
    
    Args:
        script_dir: Directory where the script is located (for finding model)
    
    Returns:
        model, device, class_names
    """
    global _model_cache, _device_cache, _class_names_cache
    
    # Try to use Streamlit cache if available
    try:
        import streamlit as st
        
        @st.cache_resource
        def _load_model_cached(_script_dir, _model_path):
            return load_model(_model_path)
        
        if script_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_dir = os.path.join(script_dir, "model")
        model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
        
        if not os.path.exists(model_path):
            best_model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
            if os.path.exists(best_model_path):
                model_path = best_model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found. Expected at: {model_path}\n"
                f"Please train the model first using main.py"
            )
        
        _model_cache, _device_cache, _class_names_cache = _load_model_cached(script_dir, model_path)
        return _model_cache, _device_cache, _class_names_cache
    
    except ImportError:
        # Streamlit not available, use simple caching
        pass
    
    # Fallback: simple global caching
    if _model_cache is not None:
        return _model_cache, _device_cache, _class_names_cache
    
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find model file
    model_dir = os.path.join(script_dir, "model")
    model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier.pth")
    
    # Try best model if main model doesn't exist
    if not os.path.exists(model_path):
        best_model_path = os.path.join(model_dir, "mobilenetv3_recyclable_classifier_best.pth")
        if os.path.exists(best_model_path):
            model_path = best_model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found. Expected at: {model_path}\n"
            f"Please train the model first using main.py"
        )
    
    _model_cache, _device_cache, _class_names_cache = load_model(model_path)
    return _model_cache, _device_cache, _class_names_cache

