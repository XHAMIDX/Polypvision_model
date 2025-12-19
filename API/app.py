import streamlit as st
import os
import sys
from openai import OpenAI
from PIL import Image
import io
import base64
import json
from typing import Optional
from dotenv import load_dotenv
import httpx
import numpy as np
import torch
import cv2
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_efficientnetv2_classifier, create_segmentation_model
from utils import get_classification_val_transform, get_segmentation_val_transform
from inference.colonoscopy_rag_database import RAG_SYSTEM_PROMPTS, get_rag_context, RAG_TRAINING_DATA

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen2.5-vl-32b-instruct:free")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Polyp Detection Chatbot")

# Load configuration for segmentation
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    st.error(f"Could not load config.yaml: {e}")
    config = {}

# Segmentation Inference Class
class SegmentationInference:
    """Simplified segmentation inference for Streamlit app"""
    
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
        try:
            checkpoint = torch.load(seg_model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.seg_model.load_state_dict(checkpoint)
            
            self.seg_model = self.seg_model.to(self.device)
            self.seg_model.eval()
            self.model_loaded = True
        except Exception as e:
            st.error(f"Could not load segmentation model: {e}")
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

# Load segmentation model in session state
@st.cache_resource
def load_segmentation_model(config):
    inference_config = config.get('inference', {})
    return SegmentationInference(
        config=inference_config,
        device=inference_config.get('device', 'cuda')
    )

# Get configuration from environment variables
API_KEY = os.getenv("OPENAI_API_KEY", "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")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "qwen/qwen2.5-vl-32b-instruct:free")
SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Polyp Detection Chatbot")

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI API client with OpenRouter
client = None
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60.0,
        default_headers={
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        }
    )
    st.session_state.api_initialized = True
except Exception as e:
    st.session_state.api_initialized = False
    st.session_state.api_error = str(e)

# Page configuration
st.set_page_config(
    page_title="Polyp Detection Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔬 Polyp Detection & Classification</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This application helps identify and classify polyp types in medical images.
    
    **Supported Polyp Types:**
    - **Hyperplastic**
    - **Adenomatous**
      - Tubular
      - Villous
      - Tubulovillous
    
    **How to use:**
    1. Upload or capture a polyp image
    2. The AI will analyze and classify the polyp type
    3. View detailed results
    """)
    
    # Configuration Section
    with st.expander("⚙️ Configuration"):
        st.subheader("OpenRouter API Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Status", "✅ Ready" if st.session_state.get('api_initialized') else "❌ Error")
        with col2:
            st.metric("Service", "OpenRouter")
        
        st.code(f"Base URL: {BASE_URL}", language="text")
        st.code(f"Model: {MODEL}", language="text")
        st.code(f"Site: {SITE_NAME}", language="text")
        
        if not st.session_state.get('api_initialized'):
            st.error(f"⚠️ API Error: {st.session_state.get('api_error', 'Unknown error')}")
            st.info("**Required Settings in .env:**\n- OPENAI_API_KEY (OpenRouter key)\n- OPENAI_BASE_URL (https://openrouter.ai/api/v1)\n- OPENAI_MODEL (meta-llama/llama-4-maverick:free)")
    
    # Debug info
    with st.expander("🔧 Debug & Test"):
        st.subheader("OpenRouter Connection Test")
        st.write(f"**API Key Configured:** {'✅' if API_KEY else '❌'}")
        st.write(f"**Base URL:** {BASE_URL}")
        st.write(f"**Model:** {MODEL}")
        st.write(f"**Site Name:** {SITE_NAME}")
        st.write(f"**Client Initialized:** {'✅ Yes' if client else '❌ No'}")
        
        if st.button("🧪 Test OpenRouter Connection", key="test_connection"):
            try:
                if not client:
                    st.error("❌ Client not initialized - Cannot test")
                else:
                    with st.spinner("Testing connection to OpenRouter API..."):
                        # Simple test message
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": "Respond with one word: 'success'"}],
                            max_tokens=20,
                            temperature=0.7
                        )
                        st.success("✅ OpenRouter Connection Successful!")
                        st.write(f"**Response:** {response.choices[0].message.content}")
                        st.balloons()
                        
                        # Show usage info if available
                        if hasattr(response, 'usage') and response.usage:
                            st.info(f"**Tokens Used:** {response.usage.total_tokens}")
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Connection Test Failed:\n\n{error_msg}")
                
                if "401" in error_msg or "UNAUTHENTICATED" in error_msg or "Unauthorized" in error_msg:
                    st.warning("**Issue: Authentication Error (401)**\n\nSolutions:\n1. Verify OpenRouter API key is correct\n2. Check if API key has expired\n3. Generate a new key from OpenRouter dashboard\n4. Ensure Bearer token format is correct")
                elif "404" in error_msg:
                    st.warning("**Issue: Not Found (404)**\n\nSolutions:\n1. Verify OpenRouter base URL is correct\n2. Check model name: `meta-llama/llama-4-maverick:free`\n3. Ensure model is available in your region")
                elif "429" in error_msg:
                    st.warning("**Issue: Rate Limited (429)**\n\nSolutions:\n1. Wait a few moments before retrying\n2. Check OpenRouter dashboard for usage limits\n3. Upgrade account if needed")
                else:
                    st.info("**Troubleshooting:**\n- Verify API credentials\n- Check internet connection\n- Visit openrouter.ai status page\n- Ensure model is available")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a polyp image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a medical image of a polyp for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🔍 Segmentation Results")
    
    if uploaded_file is not None:
        if st.button("🔬 Run Segmentation (Stage 2)", key="segment_btn"):
            try:
                # Load model
                seg_model = load_segmentation_model(config)
                
                if not seg_model.model_loaded:
                    st.error("❌ Segmentation model failed to load. Check config paths.")
                else:
                    with st.spinner("Running segmentation..."):
                        # Convert PIL image to numpy
                        image_np = np.array(Image.open(uploaded_file).convert('RGB'))
                        
                        # Run segmentation
                        mask = seg_model.segment(image_np)
                        
                        if mask is not None:
                            # Create visualization
                            overlay, mask_resized = seg_model.visualize(image_np, mask)
                            
                            # Display results in tabs
                            tab1, tab2, tab3 = st.tabs(["Overlay", "Heatmap", "Stats"])
                            
                            with tab1:
                                st.image(overlay, caption="Segmentation Overlay (Red = Detected Polyp)", use_container_width=True)
                            
                            with tab2:
                                # Create heatmap visualization
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(8, 6))
                                im = ax.imshow(mask_resized, cmap='hot')
                                ax.set_title('Segmentation Confidence Heatmap')
                                ax.axis('off')
                                plt.colorbar(im, ax=ax, label='Confidence')
                                st.pyplot(fig)
                                plt.close()
                            
                            with tab3:
                                # Calculate statistics
                                mask_binary = (mask_resized > seg_model.seg_threshold).astype(np.uint8)
                                polyp_pixels = np.sum(mask_binary)
                                total_pixels = mask_binary.size
                                polyp_percentage = (polyp_pixels / total_pixels) * 100
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Polyp Pixels", f"{polyp_pixels:,}")
                                with col_b:
                                    st.metric("Coverage %", f"{polyp_percentage:.2f}%")
                                with col_c:
                                    st.metric("Threshold", f"{seg_model.seg_threshold:.2f}")
                                
                                # Store mask in session state for classification
                                st.session_state.segmentation_mask = mask_resized
                                st.session_state.segmentation_overlay = overlay
                                st.session_state.image_array = image_np
                                
                                st.success("✅ Segmentation Complete!")
                                st.info("Now proceed to Classification for full diagnosis")
                        else:
                            st.error("❌ Segmentation failed - could not process image")
            
            except Exception as e:
                st.error(f"❌ Segmentation error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

# Classification section
st.markdown("---")
st.subheader("🤖 Classification (Stage 1)")

# Classification section
st.markdown("---")
st.subheader("🤖 Classification (Stage 1)")

if 'segmentation_overlay' in st.session_state and st.session_state.segmentation_overlay is not None:
    # Show segmentation result in classification section
    st.info("✅ Segmentation completed - displaying result above")
    
    col_seg1, col_seg2 = st.columns(2)
    with col_seg1:
        st.image(st.session_state.segmentation_overlay, caption="Segmentation Result", use_container_width=True)
    with col_seg2:
        st.markdown("### Segmentation Summary")
        mask_binary = (st.session_state.segmentation_mask > st.session_state.get('seg_threshold', 0.5)).astype(np.uint8)
        coverage = (np.sum(mask_binary) / mask_binary.size) * 100
        st.metric("Polyp Coverage", f"{coverage:.2f}%")
        st.metric("Mask Threshold", f"{st.session_state.get('seg_threshold', 0.5):.2f}")
    
    if st.button("📊 Classify with AI", key="classify_btn"):
        with st.spinner("Classifying polyp type..."):
            try:
                if not client:
                    st.error("❌ OpenRouter API not initialized. Check your API credentials.")
                else:
                    # Use the original image for classification
                    image_pil = Image.fromarray(st.session_state.image_array.astype('uint8'))
                    image_data = io.BytesIO()
                    image_pil.save(image_data, format='PNG')
                    image_data.seek(0)
                    image_base64 = base64.standard_b64encode(image_data.getvalue()).decode("utf-8")
                    
                    # Prepare the prompt with RAG context
                    system_prompt = RAG_SYSTEM_PROMPTS["polyp_classification"] + "\n\n" + RAG_SYSTEM_PROMPTS["clinical_context"]

                    user_prompt = f"""{RAG_SYSTEM_PROMPTS["image_analysis"]}

COLONOSCOPY DATABASE REFERENCE:
- Training cases reviewed: {RAG_TRAINING_DATA['total_polyps_classified']} polyps
- Adenomatous patterns: {RAG_TRAINING_DATA['adenomatous_features']['common_morphologies']}
- Hyperplastic patterns: {RAG_TRAINING_DATA['hyperplastic_features']['common_morphologies']}

Analyze this polyp image and provide classification.

You must respond ONLY with a JSON object in this exact format:
{{
    "polyp_type": "HYPERPLASTIC or ADENOMATOUS",
    "adenoma_subtype": "TUBULAR, VILLOUS, or TUBULOVILLOUS (only if ADENOMATOUS)",
    "confidence": "HIGH, MEDIUM, or LOW",
    "description": "Brief clinical description with reference to JNET types or morphology",
    "recommendations": "Clinical recommendations based on classification",
    "rag_reference": "Any relevant case from training database"
}}

If unable to classify, respond with:
{{
    "polyp_type": "UNABLE_TO_CLASSIFY",
    "confidence": "LOW",
    "description": "Reason why classification failed",
    "recommendations": "Please provide a clearer image"
}}"""

                    # Call the API
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
                    
                    # Display results
                    st.markdown(f"""
                    <div class="result-box">
                    <h3 style="color: black;">✅ Classification Complete</h3>
                                
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extract and display key information
                    try:
                        # Try to extract JSON from markdown code blocks if present
                        json_text = result_text
                        if "```json" in json_text:
                            json_text = json_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in json_text:
                            json_text = json_text.split("```")[1].split("```")[0].strip()
                        
                        result_json = json.loads(json_text)
                    
                        st.markdown("### Classification Results")
                        col_a, col_b, col_c = st.columns(3)
                    
                        with col_a:
                            st.metric("Polyp Type", result_json.get("polyp_type", "N/A"))
                        with col_b:
                            st.metric("Confidence", result_json.get("confidence", "N/A"))
                        with col_c:
                            if "adenoma_subtype" in result_json and result_json["adenoma_subtype"]:
                                st.metric("Subtype", result_json["adenoma_subtype"])
                        
                        st.markdown("### Clinical Assessment")
                        st.write(result_json.get("description", "No description available"))
                    
                        st.markdown("### Recommendations")
                        st.write(result_json.get("recommendations", "No recommendations"))
                        
                        st.success("✅ Analysis pipeline complete!")
                    
                    except json.JSONDecodeError as e:
                        st.warning(f"Could not parse JSON response: {str(e)}")
                        st.info("Raw response:")
                        st.code(result_text)
            
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

else:
    st.info("👆 Please complete segmentation first (see Segmentation Results tab)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>⚕️ Medical AI Assistant | Polyp Detection System v1.0</small><br>
    <small>Disclaimer: This tool is for informational purposes only and should not replace professional medical diagnosis.</small>
</div>
""", unsafe_allow_html=True)
