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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

class Predictor:
    """Handle model predictions for AD/NC classification."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = ['AD', 'NC']
        
        # Initialize model
        self.model = modules.convnext_medium().to(self.device)
        
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
        return image_tensor.to(self.device)
    
    def predict(self, image_path):
        with torch.no_grad():
            tensor = self.load_image(image_path)
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred, probs[0].cpu().numpy()
    
    def evaluate_directory(self, test_dir, output_path="pedictions.json"):
        y_true, y_pred = [], []
        results = []
        test_dir = Path(test_dir)

        # Collect images from subfolders named after class labels
        for class_name in self.class_names:
            class_path = test_dir / class_name
            if not class_path.exists():
                print(f"Warning: Missing class folder {class_path}")
                continue
            image_paths = list(class_path.glob("*"))
            print(f"Processing {len(image_paths)} images in {class_name}...")

            for img_path in tqdm(image_paths):
                try:
                    pred_idx, probs = self.predict(img_path)
                    y_true.append(self.class_names.index(class_name))
                    y_pred.append(pred_idx)
                    results.append({
                        'image_path': str(img_path),
                        'true_class': class_name,
                        'predicted_class': self.class_names[pred_idx],
                        'confidence_AD': float(probs[0]),
                        'confidence_NC': float(probs[1])
                    })
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        print(f"\nAccuracy: {acc:.4f}")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", json.dumps(report, indent=2))

        # Save confusion matrix as image
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix (Accuracy={acc:.2%})")
        cm_path = Path(output_path).with_suffix('.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Save all results to JSON
        output_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'total_samples': len(y_true),
            'predictions': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"Confusion matrix saved to {cm_path}")

        return acc, cm, report
    
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



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt model on AD/NC test dataset")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", default="evaluation_results.json", help="Path to output JSON")
    args = parser.parse_args()

    evaluator = Predictor(args.model)
    evaluator.evaluate_directory('ADNI/AD_NC/test', args.output)
