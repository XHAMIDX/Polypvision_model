"""
Hierarchical Inference Pipeline
1. Binary classification (Adenoma vs Hyperplastic)
2. If Adenoma → Subtype classification (Tubular, Tubulovillous, Villous)
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import argparse
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_efficientnetv2_classifier
from utils import get_classification_val_transform


class HierarchicalClassifier:
    """
    Two-stage hierarchical classifier for polyp classification
    """
    
    def __init__(
        self,
        binary_model_path: str,
        subtype_model_path: str,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Args:
            binary_model_path: Path to binary classifier checkpoint
            subtype_model_path: Path to subtype classifier checkpoint
            config: Inference configuration
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.input_size = config['input_size']
        self.binary_threshold = config.get('binary_threshold', 0.5)
        self.binary_classes = config.get('binary_classes', ['hyperplastic', 'adenoma'])
        self.subtype_classes = config.get('subtype_classes', ['tubular', 'tubulovillous', 'villous'])
        
        # Get model name from config (CRITICAL: Must match checkpoint architecture)
        # Config has 'stage1' section with 'model' key
        model_name = config.get('model', 'tf_efficientnetv2_m')
        print(f"Using model architecture: {model_name}")
        
        # Load binary model (2 classes: hyperplastic, adenoma)
        print("Loading binary classifier...")
        self.binary_model = create_efficientnetv2_classifier(
            num_classes=2, 
            pretrained=False,
            model_name=model_name
        )
        checkpoint = torch.load(binary_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.binary_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.binary_model.load_state_dict(checkpoint)
        self.binary_model = self.binary_model.to(self.device)
        self.binary_model.eval()
        
        # Load subtype model (3 classes: tubular, tubulovillous, villous)
        print("Loading subtype classifier...")
        self.subtype_model = create_efficientnetv2_classifier(
            num_classes=3, 
            pretrained=False,
            model_name=model_name
        )
        checkpoint = torch.load(subtype_model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.subtype_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.subtype_model.load_state_dict(checkpoint)
        self.subtype_model = self.subtype_model.to(self.device)
        self.subtype_model.eval()
        
        # Transform
        self.transform = get_classification_val_transform(self.input_size)
        
        # Class names
        print(f"✓ Models loaded successfully on {self.device}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transform
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict_binary(self, image_tensor: torch.Tensor) -> dict:
        """
        Binary classification: Adenoma vs Hyperplastic
        
        Args:
            image_tensor: Preprocessed image [1, 3, H, W]
        
        Returns:
            Dictionary with probabilities and prediction
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.binary_model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        return {
            'hyperplastic_prob': float(probs[0]),
            'adenoma_prob': float(probs[1]),
            'predicted_class': self.binary_classes[int(probs[1] > self.binary_threshold)]
        }
    
    def predict_subtype(self, image_tensor: torch.Tensor) -> dict:
        """
        Subtype classification: Tubular, Tubulovillous, Villous
        
        Args:
            image_tensor: Preprocessed image [1, 3, H, W]
        
        Returns:
            Dictionary with probabilities and prediction
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            logits = self.subtype_model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        result: dict[str, float | str] = {
            f'{name}_prob': float(prob) 
            for name, prob in zip(self.subtype_classes, probs)
        }
        result['predicted_subtype'] = self.subtype_classes[int(np.argmax(probs))]
        
        return result
    
    def predict(self, image_path: str) -> dict:
        """
        Hierarchical prediction
        
        Args:
            image_path: Path to image file
        
        Returns:
            Comprehensive prediction result
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        
        # Stage 1: Binary classification
        binary_result = self.predict_binary(image_tensor)
        
        # Stage 2: Subtype classification (only if adenoma)
        if binary_result['predicted_class'] == 'adenoma':
            subtype_result = self.predict_subtype(image_tensor)
            final_label = f"adenoma_{subtype_result['predicted_subtype']}"
        else:
            subtype_result = None
            final_label = 'hyperplastic'
        
        # Combine results
        result = {
            'image_path': str(image_path),
            'binary': binary_result,
            'subtype': subtype_result,
            'final_label': final_label
        }
        
        return result
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    inference_config = config['inference']
    stage1_config = config.get('stage1', {})
    
    # Add model name from stage1 to inference config
    # This ensures the correct architecture is loaded
    inference_config['model'] = config.get('stage1', {}).get('model', 'efficientnet_b2')
    inference_config.setdefault('binary_classes', ['hyperplastic', 'adenoma'])
    inference_config.setdefault(
        'subtype_classes',
        config.get('stage3', {}).get('class_names', ['tubular', 'tubulovillous', 'villous'])
    )
    
    # Create classifier
    classifier = HierarchicalClassifier(
        binary_model_path=inference_config['binary_model_path'],
        subtype_model_path=inference_config['subtype_model_path'],
        config=inference_config,
        device=inference_config['device']
    )
    
    # Process single image or directory
    if args.image:
        # Single image
        print(f"\nProcessing: {args.image}")
        result = classifier.predict(args.image)
        
        # Print result
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        
        # Save if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Saved result to {args.output}")
    
    elif args.image_dir:
        # Directory of images
        print(f"\nProcessing directory: {args.image_dir}")
        image_dir = Path(args.image_dir)
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(image_dir.glob(ext))
        
        print(f"Found {len(image_paths)} images")
        
        # Batch prediction
        results = classifier.predict_batch(image_paths)
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PREDICTION SUMMARY")
        print("="*60)
        
        hyperplastic_count = 0
        adenoma_counts = {'tubular': 0, 'tubulovillous': 0, 'villous': 0}
        
        for result in results:
            if 'error' not in result:
                if result['final_label'] == 'hyperplastic':
                    hyperplastic_count += 1
                else:
                    subtype = result['final_label'].split('_')[1]
                    adenoma_counts[subtype] += 1
        
        print(f"Hyperplastic: {hyperplastic_count}")
        print(f"Adenoma:")
        for subtype, count in adenoma_counts.items():
            print(f"  - {subtype.capitalize()}: {count}")
        
        # Save results
        output_path = args.output or 'inference_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved results to {output_path}")
    
    else:
        print("Error: Please provide --image or --image_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical Polyp Classification Inference')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    args = parser.parse_args()
    
    main(args)
