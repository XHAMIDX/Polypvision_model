"""
Refactored Inference Pipeline - Segmentation First Approach
1. Segment polyp using Stage 2 model
2. Classify polyp as Adenomatous or Hyperplastic using Stage 1 model
3. Visualize mask overlaid on original image with classification label
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
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_efficientnetv2_classifier, create_segmentation_model
from utils import get_classification_val_transform, get_segmentation_val_transform


class SegmentationFirstInference:
    """
    Two-stage inference pipeline:
    1. Segment polyp boundary (Stage 2)
    2. Classify polyp type (Stage 1)
    """
    
    def __init__(
        self,
        seg_model_path: str,
        binary_model_path: str,
        config: dict,
        device: str = 'cuda'
    ):
        """
        Args:
            seg_model_path: Path to segmentation model checkpoint
            binary_model_path: Path to binary classifier checkpoint
            config: Inference configuration
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Segmentation settings
        self.seg_input_size = config.get('seg_input_size', 512)
        self.seg_threshold = config.get('seg_threshold', 0.5)
        
        # Classification settings
        self.clf_input_size = config.get('input_size', 224)
        self.binary_threshold = config.get('binary_threshold', 0.5)
        # CRITICAL: Training uses alphabetical order of class folders, NOT hardcoded order
        # 'Adenoma' comes before 'Hyperplastic' alphabetically
        # So: index 0 = Adenoma, index 1 = Hyperplastic
        self.binary_classes = config.get('binary_classes', ['adenoma', 'hyperplastic'])
        
        print("\n" + "="*60)
        print("Loading Segmentation Model (Stage 2)...")
        print("="*60)
        
        # Load segmentation model
        seg_config = config.get('stage2', {})
        self.seg_model = create_segmentation_model(
            architecture=seg_config.get('architecture', 'UnetPlusPlus'),
            encoder_name=seg_config.get('encoder', 'efficientnet-b2'),
            pretrained_encoder=None,
            freeze_encoder=False
        )
        
        # Load checkpoint
        checkpoint = torch.load(seg_model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded segmentation checkpoint (epoch {checkpoint.get('epoch', '?')})")
        else:
            self.seg_model.load_state_dict(checkpoint)
            print("✓ Loaded segmentation checkpoint")
        
        self.seg_model = self.seg_model.to(self.device)
        self.seg_model.eval()
        
        print("\n" + "="*60)
        print("Loading Classification Model (Stage 1 - Binary)...")
        print("="*60)
        
        # Load classification model
        stage1_config = config.get('stage1', {})
        model_name = stage1_config.get('model', 'tf_efficientnetv2_m')
        
        self.clf_model = create_efficientnetv2_classifier(
            num_classes=2,
            pretrained=False,
            model_name=model_name
        )
        
        checkpoint = torch.load(binary_model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.clf_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded binary classifier checkpoint (epoch {checkpoint.get('epoch', '?')})")
        else:
            self.clf_model.load_state_dict(checkpoint)
            print("✓ Loaded binary classifier checkpoint")
        
        self.clf_model = self.clf_model.to(self.device)
        self.clf_model.eval()
        
        # Transforms
        self.seg_transform = get_segmentation_val_transform(self.seg_input_size)
        self.clf_transform = get_classification_val_transform(self.clf_input_size)
        
        print(f"\n✓ All models loaded successfully on {self.device}")
    
    def segment_polyp(self, image: np.ndarray) -> np.ndarray:
        """
        Segment polyp boundary using Stage 2 model
        
        Args:
            image: Original image as numpy array (H, W, 3) with values 0-255
        
        Returns:
            Binary segmentation mask (H, W) with values 0-1
        """
        # Apply segmentation transform
        transformed = self.seg_transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Run segmentation
        with torch.no_grad():
            logits = self.seg_model(image_tensor)
            mask = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
        
        return mask
    
    def classify_polyp(self, image: np.ndarray) -> dict:
        """
        Classify polyp as Adenomatous or Hyperplastic using Stage 1
        
        Args:
            image: Original image as numpy array (H, W, 3) with values 0-255
        
        Returns:
            Dictionary with classification results
        """
        # Apply classification transform
        transformed = self.clf_transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Run classification
        with torch.no_grad():
            logits = self.clf_model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Model outputs: probs[0]=adenoma, probs[1]=hyperplastic (alphabetical order)
        predicted_class = self.binary_classes[int(probs[1] > self.binary_threshold)]
        
        return {
            'adenoma_prob': float(probs[0]),          # index 0 = Adenoma
            'hyperplastic_prob': float(probs[1]),    # index 1 = Hyperplastic
            'predicted_class': predicted_class
        }
    
    def infer(self, image_path: str) -> dict:
        """
        Full inference pipeline
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with segmentation mask and classification result
        """
        # Load image
        image_pil = Image.open(image_path).convert('RGB')
        image = np.array(image_pil)
        original_h, original_w = image.shape[:2]
        
        print(f"\nProcessing: {image_path}")
        print(f"Original image size: {original_w} x {original_h}")
        
        # Stage 2: Segmentation
        print("  → Running segmentation...")
        mask = self.segment_polyp(image)
        
        # Resize mask back to original image size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Stage 1: Classification
        print("  → Running classification...")
        clf_result = self.classify_polyp(image)
        
        result = {
            'image_path': str(image_path),
            'original_size': [original_w, original_h],
            'segmentation': {
                'mask': mask.tolist(),
                'threshold_used': float(self.seg_threshold)
            },
            'classification': clf_result,
            'final_label': clf_result['predicted_class']
        }
        
        return result, image, mask
    
    def visualize_result(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        clf_result: dict,
        output_path: str = None,
        show: bool = True
    ) -> None:
        """
        Visualize segmentation mask overlaid on original image with classification label
        
        Args:
            image: Original image (H, W, 3) with values 0-255
            mask: Segmentation mask (H, W) with values 0-1
            clf_result: Classification result dict
            output_path: Save visualization to this path
            show: Whether to display the image
        
        Returns:
            Visualization image as numpy array
        """
        # Convert image to float for blending
        img_float = image.astype(np.float32) / 255.0
        
        # Create colored mask overlay (red for segmented area)
        overlay = img_float.copy()
        mask_binary = (mask > self.seg_threshold).astype(np.float32)
        
        # Red channel: enhance where mask is present
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_binary * 0.5, 0, 1)
        # Green and Blue: reduce where mask is present
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] - mask_binary * 0.3, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] - mask_binary * 0.3, 0, 1)
        
        # Convert back to 0-255 range
        overlay = (overlay * 255).astype(np.uint8)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Original image with overlay
        ax1.imshow(overlay)
        ax1.set_title('Segmentation Overlay', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Add classification label on the image
        predicted_class = clf_result['predicted_class']
        confidence = max(clf_result['hyperplastic_prob'], clf_result['adenoma_prob'])
        
        label_text = f"{predicted_class.upper()}\n({confidence:.1%} confidence)"
        ax1.text(
            0.05, 0.95,
            label_text,
            transform=ax1.transAxes,
            fontsize=14,
            fontweight='bold',
            color='white',
            bbox=dict(
                boxstyle='round',
                facecolor='black',
                alpha=0.7,
                edgecolor='white',
                linewidth=2
            ),
            verticalalignment='top'
        )
        
        # Right: Segmentation mask heatmap
        mask_display = ax2.imshow(mask, cmap='hot')
        ax2.set_title('Segmentation Mask (Heatmap)', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(mask_display, ax=ax2, label='Mask Confidence')
        
        # Add classification probabilities as text
        prob_text = (
            f"Hyperplastic: {clf_result['hyperplastic_prob']:.1%}\n"
            f"Adenoma: {clf_result['adenoma_prob']:.1%}"
        )
        ax2.text(
            0.05, 0.95,
            prob_text,
            transform=ax2.transAxes,
            fontsize=10,
            color='white',
            bbox=dict(
                boxstyle='round',
                facecolor='black',
                alpha=0.7,
                edgecolor='white',
                linewidth=2
            ),
            verticalalignment='top'
        )
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            fig_path = str(output_path).replace('.json', '.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved visualization to {fig_path}")
        
        # Show if requested
        if show:
            plt.show()
        
        plt.close(fig)
        
        # Return None (visualization is saved, no need to return array)
        return None


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    inference_config = config['inference']
    
    # Use best checkpoints from training
    seg_model_path = config.get('stage2', {}).get('output_model', 'checkpoints/stage2/seg_best.pth')
    binary_model_path = inference_config.get('binary_model_path', 'checkpoints/stage1/best_model.pth')
    
    print("\nInference Configuration:")
    print(f"  Segmentation model: {seg_model_path}")
    print(f"  Classification model: {binary_model_path}")
    print(f"  Device: {inference_config['device']}")
    
    # Create inference pipeline
    inference_pipeline = SegmentationFirstInference(
        seg_model_path=seg_model_path,
        binary_model_path=binary_model_path,
        config=config,
        device=inference_config['device']
    )
    
    # Process single image or directory
    if args.image:
        # Single image
        result, image, mask = inference_pipeline.infer(args.image)
        
        # Print result
        print("\n" + "="*60)
        print("INFERENCE RESULT")
        print("="*60)
        print(f"Classification: {result['final_label'].upper()}")
        print(f"  - Hyperplastic: {result['classification']['hyperplastic_prob']:.1%}")
        print(f"  - Adenoma: {result['classification']['adenoma_prob']:.1%}")
        
        # Visualize
        print("\nGenerating visualization...")
        inference_pipeline.visualize_result(
            image,
            mask,
            result['classification'],
            output_path=args.output,
            show=not args.no_display
        )
        
        # Save JSON result
        if args.output:
            json_path = str(args.output).replace('.png', '.json')
            with open(json_path, 'w') as f:
                # Don't save the full mask array to JSON (too large)
                result_to_save = result.copy()
                result_to_save['segmentation']['mask'] = "mask_not_included_in_json"
                json.dump(result_to_save, f, indent=2)
            print(f"✓ Saved result to {json_path}")
    
    elif args.image_dir:
        # Directory of images
        print(f"\nProcessing directory: {args.image_dir}")
        image_dir = Path(args.image_dir)
        image_paths = sorted([
            p for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for p in image_dir.glob(ext)
        ])
        
        print(f"Found {len(image_paths)} images")
        
        # Process all images
        results = []
        adenoma_count = 0
        hyperplastic_count = 0
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"\n[{i}/{len(image_paths)}] Processing...")
                result, image, mask = inference_pipeline.infer(str(image_path))
                results.append(result)
                
                if result['final_label'] == 'adenoma':
                    adenoma_count += 1
                else:
                    hyperplastic_count += 1
                
                # Visualize each image
                output_name = image_path.stem
                output_base = args.output or 'inference_results'
                output_path = Path(output_base) / f"{output_name}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                inference_pipeline.visualize_result(
                    image,
                    mask,
                    result['classification'],
                    output_path=str(output_path),
                    show=False
                )
                
            except Exception as e:
                print(f"✗ Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images: {len(image_paths)}")
        print(f"Adenoma: {adenoma_count}")
        print(f"Hyperplastic: {hyperplastic_count}")
        
        # Save results
        output_base = args.output or 'inference_results'
        json_output = Path(output_base) / 'results.json'
        json_output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_output, 'w') as f:
            # Remove mask arrays from JSON
            results_to_save = []
            for r in results:
                r_copy = r.copy()
                if 'segmentation' in r_copy and 'mask' in r_copy['segmentation']:
                    r_copy['segmentation']['mask'] = "mask_not_included_in_json"
                results_to_save.append(r_copy)
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n✓ Saved results to {json_output}")
    
    else:
        print("Error: Please provide --image or --image_dir")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Segmentation-First Polyp Inference Pipeline'
    )
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output path for visualizations')
    parser.add_argument('--no-display', action='store_true', help='Do not display images')
    args = parser.parse_args()
    
    main(args)
