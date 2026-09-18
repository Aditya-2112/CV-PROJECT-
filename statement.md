# Problem Statement

Traffic sign recognition is a core perception task for autonomous vehicles and Advanced
Driver-Assistance Systems (ADAS). Given an image of a traffic sign, a system must correctly
classify it into its category (speed limit, warning, prohibitory, mandatory, etc.), since
misclassification can have direct real-world safety consequences.

This project builds a single-image, multi-class traffic sign classifier trained from scratch
using a custom Convolutional Neural Network (CNN), on the German Traffic Sign Recognition
Benchmark (GTSRB) dataset — 43 sign classes, 50,000+ real-world images.

# Scope

**In scope:**
- Classification of a single, pre-cropped traffic sign image into one of 43 classes
- Model training, validation, and checkpointing pipeline
- Evaluation on a held-out test set with quantitative metrics (accuracy, confusion matrix)
- Command-line interface for training and inference (single image, folder, or full test set)

**Out of scope:**
- Sign detection/localization within a full driving scene (this project assumes signs are
  already cropped/isolated)
- Real-time video inference
- Deployment as a web service or embedded system

# Target Users

- Computer vision / ML students and instructors evaluating a from-scratch CNN classification pipeline
- Developers prototyping a traffic-sign recognition component for a larger ADAS or mapping pipeline
- Researchers benchmarking simple custom CNN architectures against the GTSRB dataset

# High-Level Features

1. **Training module** — loads and augments GTSRB data, trains a custom 4-block CNN with
   batch normalization and dropout, tracks train/validation loss and accuracy per epoch, and
   checkpoints the best-performing model.
2. **Prediction module** — loads a trained checkpoint and classifies a single image, a folder
   of images, or the entire official test set, printing predicted class and confidence.
3. **Evaluation module** — computes test-set accuracy and generates a confusion matrix to
   analyze per-class performance and common misclassifications.
4. **CLI interface** — a single entry point (`main.py`) exposing all the above via `argparse`,
   fully runnable from a terminal with no GUI dependency.
