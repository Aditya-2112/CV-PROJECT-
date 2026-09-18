import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns # Used for nicer confusion matrix plotting (optional but good practice)
from tqdm import tqdm

from model import TrafficSignCNN
from utils import get_transforms, get_dataloaders, get_device, GTSRB_CLASSES

def load_model(model_path, device):
    """Loads a trained model from a checkpoint path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    model = TrafficSignCNN(num_classes=43)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle whether we saved full checkpoint or just state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval() # Important: set to evaluation mode
    return model

def predict_single_image(model, image_path, device):
    """Runs inference on a single image file."""
    # Preprocessing transform (no augmentation)
    transform = get_transforms(is_train=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None
        
    # Add batch dimension: shape becomes (1, 3, 32, 32)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, 0)
        
    class_id = predicted_idx.item()
    conf_percent = confidence.item() * 100
    class_name = GTSRB_CLASSES.get(class_id, "Unknown")
    
    return class_name, conf_percent

def predict_folder(model, folder_path, device):
    """Runs inference on all common image files in a directory."""
    extensions = ('*.png', '*.jpg', '*.jpeg')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_paths:
        print(f"No images found in {folder_path}")
        return
        
    print(f"Found {len(image_paths)} images. Running inference...\n")
    for img_path in image_paths:
        class_name, conf = predict_single_image(model, img_path, device)
        if class_name:
            basename = os.path.basename(img_path)
            print(f"{basename}: {class_name} ({conf:.1f}%)")

def evaluate_test_set(model, data_dir, output_dir, device):
    """Evaluates the model on the full GTSRB test set and saves a confusion matrix."""
    print("Loading test dataset...")
    # Batch size can be larger for inference since no gradients are stored
    _, _, test_loader = get_dataloaders(data_dir, batch_size=128)
    
    all_preds = []
    all_labels = []
    
    print("Running predictions on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate accuracy
    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\nTest Set Accuracy: {acc:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 12)) # Make it large enough for 43 classes
    # Use seaborn for a nice heatmap
    try:
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='d',
                    xticklabels=range(43), yticklabels=range(43))
    except NameError:
        # Fallback if seaborn fails for some reason
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        
    plt.title(f'GTSRB Test Set Confusion Matrix (Acc: {acc:.2f}%)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

def run_prediction(model_path, image_target, data_dir, output_dir='outputs'):
    """Main prediction orchestrator."""
    device = get_device()
    print(f"Using device: {device}")
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    if image_target:
        # Check if it's a file or directory
        if os.path.isfile(image_target):
            class_name, conf = predict_single_image(model, image_target, device)
            print(f"Prediction: {class_name} (Confidence: {conf:.1f}%)")
        elif os.path.isdir(image_target):
            predict_folder(model, image_target, device)
        else:
            print(f"Error: {image_target} is neither a file nor a directory.")
    elif data_dir:
        # If no specific image provided but data_dir is, evaluate on test set
        evaluate_test_set(model, data_dir, output_dir, device)
    else:
        print("Please provide either --image or --data-dir for prediction.")
