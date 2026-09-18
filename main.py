import argparse
from train import train_model
from predict import run_prediction

def main():
    """
    Main entry point for the GTSRB Traffic Sign Classifier CLI.
    Parses command line arguments and routes to either training or prediction.
    """
    parser = argparse.ArgumentParser(description="GTSRB Traffic Sign Classifier CLI")
    
    # Primary mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'],
                        help="Mode to run the script in: 'train' or 'predict'")
    
    # Common arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help="Path to dataset directory. Will download here if missing.")
    parser.add_argument('--model-path', type=str, default='outputs/best_model.pth',
                        help="Path to save (train) or load (predict) the model checkpoint.")
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help="Directory to save plots and checkpoints.")
                        
    # Training-specific arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of epochs to train (default: 20)")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="Batch size for training/inference (default: 64)")
                        
    # Prediction-specific arguments
    parser.add_argument('--image', type=str, default=None,
                        help="Path to a single image or folder of images for inference.")
                        
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("--- Starting Training Mode ---")
        train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
    elif args.mode == 'predict':
        print("--- Starting Prediction Mode ---")
        run_prediction(
            model_path=args.model_path,
            image_target=args.image,
            data_dir=args.data_dir if args.image is None else None, # Evaluate on test set if no image provided
            output_dir=args.output_dir
        )

if __name__ == '__main__':
    main()
