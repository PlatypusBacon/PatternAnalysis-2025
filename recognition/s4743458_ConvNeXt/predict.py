import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import modules
from torchvision import transforms
from tqdm import tqdm
import json
from datetime import datetime

class Predictor:
    """Handle model predictions for AD/NC classification."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = ['AD', 'NC']
        
        # Initialize model
        self.model = modules.convnext_small().to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def load_checkpoint(self, checkpoint_path):

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print checkpoint info if available
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"Model trained to epoch {checkpoint.get('epoch', 'N/A')}")
            print(f"Validation Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"Validation F1: {metrics.get('f1', 'N/A'):.4f}")
        print(f"Model loaded successfully on {self.device}\n")
    
    def load_image(self, image_path):
        """Load and preprocess a single image."""
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, image
    
    def predict_single(self, image_path, return_probs=True):
        """Predict on a single image."""
        image_tensor, original_image = self.load_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        result = {
            'predicted_class': self.class_names[pred_class],
            'predicted_index': pred_class,
            'confidence': confidence,
            'probabilities': {
                'AD': probs[0, 0].item(),
                'NC': probs[0, 1].item()
            }
        }
        
        if return_probs:
            return result, original_image
        return result
    
    def predict_batch(self, image_paths):
        """Predict on multiple images."""
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for image_path in tqdm(image_paths):
            try:
                result = self.predict_single(image_path, return_probs=False)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path, recursive=False):
        """Predict on all images in a directory."""
        directory = Path(directory_path)
        
        # Supported image extensions
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        # Find all images
        if recursive:
            image_paths = []
            for ext in extensions:
                image_paths.extend(directory.rglob(f'*{ext}'))
        else:
            image_paths = []
            for ext in extensions:
                image_paths.extend(directory.glob(f'*{ext}'))
        
        if not image_paths:
            print(f"No images found in {directory_path}")
            return []
        
        return self.predict_batch(image_paths)
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with confidence scores."""
        result, original_image = self.predict_single(image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show original image
        axes[0].imshow(original_image, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f'Input Image\n{Path(image_path).name}')
        
        # Show prediction probabilities
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors = ['#FF6B6B' if c == result['predicted_class'] else '#4ECDC4' for c in classes]
        
        axes[1].barh(classes, probs, color=colors)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('Probability')
        axes[1].set_title(f'Prediction: {result["predicted_class"]}\n'
                         f'Confidence: {result["confidence"]:.2%}')
        
        # Add probability values on bars
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            axes[1].text(prob + 0.02, i, f'{prob:.2%}', 
                        va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result
    
    def save_results(self, results, output_path):
        """Save prediction results to JSON file."""
        output_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_images': len(results),
            'predictions': results
        }
        
        # Calculate summary statistics
        successful_preds = [r for r in results if 'error' not in r]
        if successful_preds:
            class_counts = {}
            for result in successful_preds:
                cls = result['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            output_data['summary'] = {
                'successful_predictions': len(successful_preds),
                'failed_predictions': len(results) - len(successful_preds),
                'class_distribution': class_counts,
                'average_confidence': np.mean([r['confidence'] for r in successful_preds])
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        if 'summary' in output_data:
            print("\nSummary:")
            print(f"  Successful predictions: {output_data['summary']['successful_predictions']}")
            print(f"  Failed predictions: {output_data['summary']['failed_predictions']}")
            print(f"  Class distribution: {output_data['summary']['class_distribution']}")
            print(f"  Average confidence: {output_data['summary']['average_confidence']:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Predict AD/NC classification using trained ConvNeXt model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to directory for output')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--directory', type=str, default=None,
                       help='Path to directory containing images')
    args = parser.parse_args()
    
    # Validate inputs
    if args.image is None and args.directory is None:
        parser.error("Either --image or --directory must be specified")
    # Initialize predictor
    predictor = Predictor(args.model, device='cuda')
    
    # Single image prediction
    if args.image:
        print(f"Predicting on single image: {args.image}\n")
        
        if args.output:
            result = predictor.visualize_prediction(args.image, save_path=args.output)
        else:
            result = predictor.predict_single(args.image, return_probs=False)
        
        print("\nPrediction Result:")
        print(f"  Image: {args.image}")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"    {cls}: {prob:.2%}")
    
    # Directory prediction
    elif args.directory:
        print(f"Predicting on directory: {args.directory}")
        
        results = predictor.predict_directory(args.directory)
        
        if results:
            predictor.save_results(results, args.output)
        else:
            print("No predictions made.")


if __name__ == "__main__":
    main()
