from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import torch
import yaml
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import base64
import json
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'APIcode', 'Polypvision_model'))

from models import create_efficientnetv2_classifier, create_segmentation_model
from utils import get_classification_val_transform, get_segmentation_val_transform
from inference.colonoscopy_rag_database import RAG_SYSTEM_PROMPTS, get_rag_context, RAG_TRAINING_DATA

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen2.5-vl-32b-instruct:free")

# Load configuration for models
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'APIcode', 'Polypvision_model', 'config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"Could not load config.yaml: {e}")
    config = {}

# Model cache to avoid reloading
model_cache = {}

# Segmentation Inference Class
class SegmentationInference:
    """Simplified segmentation inference for API"""

    def __init__(self, config: dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.seg_input_size = config.get('seg_input_size', 512)
        self.seg_threshold = config.get('seg_threshold', 0.5)

        # Load segmentation model
        seg_config = config.get('stage2', {})
        self.seg_model = create_segmentation_model(
            architecture=seg_config.get('architecture', 'UnetPlusPlus'),
            encoder_name=seg_config.get('encoder', 'efficientnet-b2'),
            pretrained_encoder=None,
            freeze_encoder=False
        )

        # Load checkpoint
        seg_model_path = seg_config.get('output_model', 'checkpoints/stage2/seg_best.pth')
        # Convert relative path to absolute path based on the APIcode directory
        abs_seg_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), seg_model_path)

        try:
            checkpoint = torch.load(abs_seg_model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.seg_model.load_state_dict(checkpoint)

            self.seg_model = self.seg_model.to(self.device)
            self.seg_model.eval()
            self.model_loaded = True
            print(f"✓ Segmentation model loaded from: {abs_seg_model_path}")
        except Exception as e:
            print(f"Could not load segmentation model: {e}")
            self.model_loaded = False

        self.seg_transform = get_segmentation_val_transform(self.seg_input_size)

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Run segmentation on image"""
        if not self.model_loaded:
            return None

        # Apply transform
        transformed = self.seg_transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Run segmentation
        with torch.no_grad():
            logits = self.seg_model(image_tensor)
            mask = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        return mask

    def visualize(self, image: np.ndarray, mask: np.ndarray):
        """Create visualization of segmentation result"""
        if mask is None:
            return None

        # Resize mask to original image size
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Create overlay
        img_float = image.astype(np.float32) / 255.0
        overlay = img_float.copy()
        mask_binary = (mask > self.seg_threshold).astype(np.float32)

        # Apply red overlay where mask is detected
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + mask_binary * 0.5, 0, 1)
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] - mask_binary * 0.3, 0, 1)
        overlay[:, :, 2] = np.clip(overlay[:, :, 2] - mask_binary * 0.3, 0, 1)

        overlay = (overlay * 255).astype(np.uint8)

        return overlay, mask


# Classification Inference Class
class ClassificationInference:
    """Binary classification inference for Hyperplastic/Adenomatous polyps"""

    def __init__(self, config: dict, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.clf_input_size = config.get('input_size', 224)
        self.binary_threshold = config.get('binary_threshold', 0.5)
        self.binary_classes = config.get('binary_classes', ['adenoma', 'hyperplastic'])

        # Load classification model - must match training architecture from config.yaml
        stage1_config = config.get('stage1', {})
        # CRITICAL: Use efficientnet_b2 to match the trained checkpoint (config.yaml stage1.model)
        model_name = stage1_config.get('model', 'efficientnet_b2')

        self.clf_model = create_efficientnetv2_classifier(
            num_classes=2,
            pretrained=False,
            model_name=model_name
        )

        # Use backbone_pretrained_onA.pth which matches the trained model architecture
        binary_model_path = stage1_config.get('output_path', 'checkpoints/stage1/backbone_pretrained_onA.pth')
        # Convert relative path to absolute path based on the project directory
        abs_clf_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), binary_model_path)

        try:
            checkpoint = torch.load(abs_clf_model_path, map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load with strict=False to handle minor architecture differences
            self.clf_model.load_state_dict(state_dict, strict=False)

            self.clf_model = self.clf_model.to(self.device)
            self.clf_model.eval()
            self.model_loaded = True
            print(f"✓ Classification model loaded: {model_name}")
        except Exception as e:
            print(f"Could not load classification model: {e}")
            self.model_loaded = False

        self.clf_transform = get_classification_val_transform(self.clf_input_size)

    def classify(self, image: np.ndarray) -> dict:
        """Run classification on image"""
        if not self.model_loaded:
            return None

        # Apply transform
        transformed = self.clf_transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)

        # Run classification
        with torch.no_grad():
            import torch.nn.functional as F
            logits = self.clf_model(image_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Reversed mapping: probs[0]=adenoma, probs[1]=hyperplastic
        adenoma_prob = float(probs[0])
        hyperplastic_prob = float(probs[1])

        # Predict: if probs[1] (hyperplastic) is higher, predict hyperplastic, else adenoma
        predicted_class = 'hyperplastic' if hyperplastic_prob > adenoma_prob else 'adenoma'

        return {
            'adenoma_prob': adenoma_prob,
            'hyperplastic_prob': hyperplastic_prob,
            'predicted_class': predicted_class,
            'confidence': float(max(adenoma_prob, hyperplastic_prob))
        }


def load_models():
    """Load all models once and store in cache"""
    global model_cache

    # Load configuration for inference
    inference_config = config.get('inference', {})

    if 'segmentation_model' not in model_cache:
        model_cache['segmentation_model'] = SegmentationInference(
            config=config,
            device=inference_config.get('device', 'cuda')
        )

    if 'classification_model' not in model_cache:
        model_cache['classification_model'] = ClassificationInference(
            config=config,
            device=inference_config.get('device', 'cuda')
        )


def get_client():
    """Initialize OpenAI client for RAG model"""
    try:
        # Check if API key is provided
        if not API_KEY or API_KEY == "":
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return None

        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=60.0,
            # Add headers for OpenRouter if using that service
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME", "PolypVision API"),
            }
        )
        return client
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None


@app.get("/")
def health_check():
    return {"health_check": "OK I am running on my own!"}


@app.get("/info")
def info():
    return {
        "name": "PolypVision back of back",
        "description": "you can classify, segment and then find the opinion of an expert!"
    }


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify polyp as hyperplastic or adenomatous"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Load models if not already loaded
        if not model_cache:
            load_models()

        # Check if classification model is loaded
        if 'classification_model' not in model_cache:
            raise HTTPException(status_code=500, detail="Classification model not loaded")

        clf_model = model_cache['classification_model']
        if not clf_model.model_loaded:
            raise HTTPException(status_code=500, detail="Classification model failed to load")

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))

        # Run classification
        result = clf_model.classify(image_np)

        if result is None:
            raise HTTPException(status_code=500, detail="Classification failed")

        return {
            "classification": "success",
            "result": result
        }

    except Exception as e:
        print(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    """Segment polyp from the image"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Load models if not already loaded
        if not model_cache:
            load_models()

        # Check if segmentation model is loaded
        if 'segmentation_model' not in model_cache:
            raise HTTPException(status_code=500, detail="Segmentation model not loaded")

        seg_model = model_cache['segmentation_model']
        if not seg_model.model_loaded:
            raise HTTPException(status_code=500, detail="Segmentation model failed to load")

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))

        # Run segmentation
        mask = seg_model.segment(image_np)

        if mask is None:
            raise HTTPException(status_code=500, detail="Segmentation failed")

        # Create visualization
        overlay, mask_resized = seg_model.visualize(image_np, mask)

        # Calculate statistics
        mask_binary = (mask_resized > seg_model.seg_threshold).astype(np.uint8)
        polyp_pixels = np.sum(mask_binary)
        total_pixels = mask_binary.size
        polyp_percentage = (polyp_pixels / total_pixels) * 100

        # Encode overlay image as base64
        overlay_pil = Image.fromarray(overlay.astype('uint8'))
        overlay_buffer = io.BytesIO()
        overlay_pil.save(overlay_buffer, format='PNG')
        overlay_buffer.seek(0)
        overlay_base64 = base64.standard_b64encode(overlay_buffer.getvalue()).decode("utf-8")

        return {
            "segmentation": "success",
            "polyp_coverage_percent": float(polyp_percentage),
            "polyp_pixels": int(polyp_pixels),
            "overlay_image": f"data:image/png;base64,{overlay_base64}"
        }

    except Exception as e:
        print(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/expert")
async def expert_opinion(file: UploadFile = File(...)):
    """Get expert opinion combining classification, segmentation and RAG model"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Load models if not already loaded
        if not model_cache:
            load_models()

        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        image_pil = Image.fromarray(image_np.astype('uint8'))

        # Run classification
        clf_model = model_cache['classification_model']
        if not clf_model.model_loaded:
            raise HTTPException(status_code=500, detail="Classification model failed to load")

        classification_result = clf_model.classify(image_np)

        # Run segmentation
        seg_model = model_cache['segmentation_model']
        if not seg_model.model_loaded:
            raise HTTPException(status_code=500, detail="Segmentation model failed to load")

        segmentation_mask = seg_model.segment(image_np)

        if segmentation_mask is None:
            raise HTTPException(status_code=500, detail="Segmentation failed")

        # Calculate segmentation stats
        mask_binary = (segmentation_mask > seg_model.seg_threshold).astype(np.uint8)
        polyp_pixels = np.sum(mask_binary)
        total_pixels = mask_binary.size
        polyp_coverage = (polyp_pixels / total_pixels) * 100

        # Prepare image for OpenAI API
        image_data = io.BytesIO()
        image_pil.save(image_data, format='PNG')
        image_data.seek(0)
        image_base64 = base64.standard_b64encode(image_data.getvalue()).decode("utf-8")

        # Initialize OpenAI client
        client = get_client()
        if not client:
            # Return only the local model results if API is not available
            return {
                "expert_opinion": "success_with_local_models_only",
                "local_classification": classification_result,
                "segmentation_stats": {
                    "polyp_coverage_percent": float(polyp_coverage),
                    "polyp_pixels": int(polyp_pixels),
                    "total_pixels": int(total_pixels)
                },
                "ai_refined_result": {
                    "polyp_type": "NO_API_CONNECTION",
                    "confidence": "LOW",
                    "description": "Could not connect to the external AI service for refined analysis. Using local models only.",
                    "recommendations": "Please check your API configuration and connection",
                    "agreement_with_local_model": "No external model to compare with"
                },
                "warning": "API connection not available - results based on local models only"
            }

        # Prepare prompt for RAG model
        system_prompt = RAG_SYSTEM_PROMPTS["polyp_classification"] + "\n\n" + RAG_SYSTEM_PROMPTS["clinical_context"]

        user_prompt = f"""{RAG_SYSTEM_PROMPTS["image_analysis"]}

LOCAL MODEL ANALYSIS (Stage 1 Binary Classifier):
- Predicted: {classification_result['predicted_class'].upper()}
- Adenomatous probability: {classification_result['adenoma_prob']:.1%}
- Hyperplastic probability: {classification_result['hyperplastic_prob']:.1%}
- Confidence: {classification_result['confidence']:.1%}

SEGMENTATION RESULTS:
- Polyp coverage: {polyp_coverage:.2f}%
- Polyp pixels: {polyp_pixels:,} out of {total_pixels:,} total pixels
- Threshold: {seg_model.seg_threshold:.2f}

COLONOSCOPY DATABASE REFERENCE:
- Training cases reviewed: {RAG_TRAINING_DATA['total_polyps_classified']} polyps
- Adenomatous patterns: {RAG_TRAINING_DATA['adenomatous_features']['common_morphologies']}
- Hyperplastic patterns: {RAG_TRAINING_DATA['hyperplastic_features']['common_morphologies']}

Based on the local model analysis and the image, provide refined classification and clinical assessment.

You must respond ONLY with a JSON object in this exact format:
{{
    "polyp_type": "HYPERPLASTIC or ADENOMATOUS",
    "adenoma_subtype": "TUBULAR, VILLOUS, or TUBULOVILLOUS (only if ADENOMATOUS)",
    "confidence": "HIGH, MEDIUM, or LOW",
    "description": "Brief clinical description with reference to JNET types or morphology",
    "recommendations": "Clinical recommendations based on classification",
    "agreement_with_local_model": "Whether AI agrees or disagrees with local model prediction",
    "rag_reference": "Any relevant case from training database"
}}

If unable to classify, respond with:
{{
    "polyp_type": "UNABLE_TO_CLASSIFY",
    "confidence": "LOW",
    "description": "Reason why classification failed",
    "recommendations": "Please provide a clearer image"
}}"""

        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )

            # Parse response
            result_text = response.choices[0].message.content

            # Attempt to extract JSON
            try:
                # Try to extract JSON from markdown code blocks if present
                json_text = result_text
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0].strip()

                result_json = json.loads(json_text)
            except json.JSONDecodeError:
                # If parsing fails, return raw response
                result_json = {
                    "polyp_type": "PARSING_ERROR",
                    "confidence": "LOW",
                    "description": "Could not parse the AI response",
                    "recommendations": "Internal error in processing",
                    "raw_response": result_text
                }
        except Exception as api_error:
            print(f"API call error: {api_error}")
            # Return local results with API error information
            result_json = {
                "polyp_type": "API_ERROR",
                "confidence": "LOW",
                "description": f"API service error occurred: {str(api_error)}",
                "recommendations": "Please check your API credentials and connection",
                "api_error_details": str(api_error)
            }

        # Return combined result
        return {
            "expert_opinion": "success",
            "local_classification": classification_result,
            "segmentation_stats": {
                "polyp_coverage_percent": float(polyp_coverage),
                "polyp_pixels": int(polyp_pixels),
                "total_pixels": int(total_pixels)
            },
            "ai_refined_result": result_json
        }

    except Exception as e:
        print(f"Expert opinion error: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Expert opinion failed: {str(e)}")


if __name__ == "__main__":
    # Initialize models when the server starts
    load_models()
    print("Models loaded. Server starting...")