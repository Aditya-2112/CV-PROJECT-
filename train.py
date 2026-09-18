import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import TrafficSignCNN
from utils import get_dataloaders, get_device

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plots training and validation metrics and saves the figure."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, label='Train Accuracy')
    ax2.plot(epochs, val_accs, label='Val Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, loader, criterion, optimizer, device):
    """Trains the model for one epoch and returns average loss and accuracy."""
    model.train() # Set model to training mode (enables dropout, batchnorm updates)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm adds a nice progress bar
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() # Clear gradients from previous step
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass (compute gradients)
        optimizer.step() # Update weights
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1) # Get the index of the max log-probability
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval() # Set model to evaluation mode (disables dropout, uses population stats for batchnorm)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # torch.no_grad() disables gradient calculation, saving memory and compute
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_model(data_dir, epochs=20, batch_size=64, output_dir='outputs'):
    """Main training orchestrator."""
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)
    
    # 2. Initialize model, loss, optimizer, and scheduler
    model = TrafficSignCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer is robust and generally converges faster than SGD
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # StepLR decays learning rate by a factor of 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Tracking metrics
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    # 3. Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 20)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Step the scheduler to adjust learning rate if needed
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, best_model_path)
            print(f"-> Best model saved to {best_model_path} (Acc: {val_acc:.2f}%)")
            
    # 4. Save final plots
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plot_metrics(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        plot_path
    )
    print(f"\nTraining complete! Curves saved to {plot_path}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
