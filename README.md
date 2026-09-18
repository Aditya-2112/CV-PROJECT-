# GTSRB Traffic Sign Classifier

A modular, command-line Python project for traffic sign image classification using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. This project implements a custom Convolutional Neural Network (CNN) in PyTorch, trained from scratch.

## Features

- Train a custom CNN from scratch on the GTSRB traffic sign dataset
- Predict the class of a single traffic sign image
- Predict classes for a folder of images in batch
- Evaluate model accuracy on the full held-out test set
- Automatically generate training curves and a confusion matrix for analysis
- Fully runnable from the command line, no GUI required

## Technologies Used

- Python 3.12
- PyTorch (torch, torchvision) — model definition, training, and data loading
- Matplotlib — training curve and confusion matrix visualization
- Git & GitHub — version control

## Project Structure

- `main.py`: The Command Line Interface (CLI) entry point using `argparse`. Routes to training or prediction.
- `model.py`: Defines the `TrafficSignCNN` architecture. Uses a 4-block standard CNN with BatchNorm, MaxPool, AdaptiveAvgPool, and Dropout.
- `train.py`: Contains the training loop, validation logic, and metric plotting. Saves the best model based on validation accuracy.
- `predict.py`: Handles loading the saved model and running inference on a single image, a folder of images, or evaluating the entire test set (with a confusion matrix).
- `utils.py`: Manages data downloading (via `torchvision.datasets`), data splitting, and data augmentation/preprocessing transforms.

## Requirements

The project dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

*Note: The requirements use minimum bounds (`>=`) to prevent conflicts with your specific CUDA/Python environment.*

## Usage

Use `main.py` to run the project. Data will be automatically downloaded to the `./data` directory on the first run. Checkpoints and plots are saved to `./outputs`.

### Training

To train the model from scratch:

```bash
python main.py --mode train --data-dir ./data --epochs 15 --batch-size 64
```
This will train the CNN, evaluate on a 20% validation split after each epoch, save `outputs/best_model.pth`, and generate `outputs/training_curves.png`.

### Prediction / Inference

**1. Predict a single image:**
```bash
python main.py --mode predict --model-path outputs/best_model.pth --image path/to/your/image.png
```

**2. Predict on a folder of images:**
```bash
python main.py --mode predict --model-path outputs/best_model.pth --image path/to/folder/
```

**3. Evaluate on the entire test set (Generates Confusion Matrix):**
```bash
python main.py --mode predict --model-path outputs/best_model.pth --data-dir ./data
```
This evaluates the model on the official GTSRB test set and saves `outputs/confusion_matrix.png`.

## Architecture Decisions

The model (`model.py`) is designed to be easily understandable for evaluation:
- **No Pretrained Backbone**: Built from scratch to demonstrate fundamental CNN concepts.
- **Progressive Feature Extraction**: Channel depth doubles (32->64->128->256) across 4 layers, standard practice for hierarchical feature learning.
- **Batch Normalization**: Applied after convolutions to stabilize learning and allow faster convergence.
- **Adaptive Average Pooling**: Reduces the spatial dimensions to 1x1 before the fully connected layer. This makes the network robust and removes the need to calculate precise flatten sizes based on input resolution.
- **Dropout (0.5)**: Applied before the final classifier to prevent overfitting.
