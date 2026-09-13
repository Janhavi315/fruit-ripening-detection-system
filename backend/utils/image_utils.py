"""
utils/image_utils.py - Image Utility Functions
Helper functions for image loading, validation, encoding, and preprocessing.
"""

import cv2
import numpy as np
import base64
import os
from PIL import Image
import io


# ─── Allowed file types ────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_IMAGE_SIZE_MB = 10


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from filepath using OpenCV.
    Handles various formats and returns BGR numpy array.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    img = cv2.imread(filepath)

    if img is None:
        # Try PIL as fallback (handles more formats)
        pil_img = Image.open(filepath).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if img is None:
        raise ValueError(f"Could not load image: {filepath}")

    return img


def validate_image(filepath: str) -> tuple:
    """
    Validate an uploaded image file.
    Returns (is_valid: bool, error_message: str)
    """
    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, f"Image too large ({size_mb:.1f}MB). Max allowed: {MAX_IMAGE_SIZE_MB}MB"

    # Try loading the image
    try:
        img = load_image(filepath)
        if img.shape[0] < 50 or img.shape[1] < 50:
            return False, "Image too small. Minimum 50x50 pixels required."
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def image_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """
    Convert a numpy image array to base64 encoded string.
    Used for sending annotated images back to frontend.
    """
    # Encode to bytes
    success, buffer = cv2.imencode(
        f'.{format.lower()}',
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 85] if format.upper() == 'JPEG' else []
    )

    if not success:
        raise ValueError("Failed to encode image to bytes")

    # Convert to base64 string
    img_bytes = buffer.tobytes()
    b64_str = base64.b64encode(img_bytes).decode('utf-8')

    return f"data:image/{format.lower()};base64,{b64_str}"


def resize_for_display(image: np.ndarray, max_dim: int = 600) -> np.ndarray:
    """
    Resize image for display while maintaining aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_for_analysis(image: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing pipeline for fruit analysis:
    - Resize to analysis size
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better color
    - Slight denoise
    """
    # Resize
    img = cv2.resize(image, (400, 400))

    # Convert to LAB for CLAHE (works better than direct RGB CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE to L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_channel)

    # Merge back and convert to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Slight denoise
    img_denoised = cv2.fastNlMeansDenoisingColored(img_enhanced, None, 5, 5, 7, 21)

    return img_denoised


def get_image_stats(image: np.ndarray) -> dict:
    """
    Calculate basic image statistics for debugging/logging.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    return {
        'width': image.shape[1],
        'height': image.shape[0],
        'channels': image.shape[2] if len(image.shape) > 2 else 1,
        'mean_hue': round(float(np.mean(h)), 2),
        'mean_saturation': round(float(np.mean(s)), 2),
        'mean_brightness': round(float(np.mean(v)), 2),
        'std_hue': round(float(np.std(h)), 2)
    }
