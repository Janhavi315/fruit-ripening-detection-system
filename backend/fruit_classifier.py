"""
fruit_classifier.py - Fruit Type Detection Module
Uses HSV color space analysis to identify fruit type (Mango / Banana / Apple)
Traditional CV approach: color histogram + HSV range masking
"""

import cv2
import numpy as np


# ─── HSV Color Ranges for Each Fruit ──────────────────────────────────────────
# HSV: Hue (0-179), Saturation (0-255), Value (0-255) in OpenCV
# These ranges capture the dominant colors associated with each fruit

FRUIT_COLOR_PROFILES = {
    'Mango': {
        # Mango: Yellow-orange to greenish-yellow hues
        'ranges': [
            # Yellow-orange range (ripe mango)
            (np.array([15, 80, 100]), np.array([35, 255, 255])),
            # Green-yellow range (raw mango)
            (np.array([35, 40, 80]), np.array([75, 255, 220])),
        ],
        'min_coverage': 0.12   # At least 12% of image must match
    },
    'Banana': {
        # Banana: Bright yellow to brownish-yellow
        'ranges': [
            # Bright yellow (fresh banana)
            (np.array([20, 100, 150]), np.array([35, 255, 255])),
            # Greenish-yellow (raw banana)
            (np.array([35, 60, 100]), np.array([55, 255, 230])),
        ],
        'min_coverage': 0.10
    },
    'Apple': {
        # Apple: Red (two HSV ranges because red wraps around 0/180)
        # + green for Granny Smith
        'ranges': [
            # Red range 1 (hue near 0)
            (np.array([0, 80, 80]), np.array([10, 255, 255])),
            # Red range 2 (hue near 180)
            (np.array([160, 80, 80]), np.array([179, 255, 255])),
            # Green apple range
            (np.array([40, 50, 80]), np.array([80, 255, 200])),
        ],
        'min_coverage': 0.08
    }
}


def classify_fruit(image: np.ndarray) -> dict:
    """
    Classify the fruit in the image using HSV color analysis.

    Algorithm:
    1. Convert image to HSV color space
    2. For each fruit, apply color masks and measure pixel coverage
    3. Score each fruit by weighted coverage
    4. Return the fruit with highest confidence score

    Args:
        image: BGR image array (from cv2.imread)

    Returns:
        dict with fruit_type, confidence, and color_scores
    """

    # ── Step 1: Preprocess Image ───────────────────────────────────────────
    # Resize for consistent analysis (300x300 pixels)
    img_resized = cv2.resize(image, (300, 300))

    # Apply slight Gaussian blur to reduce noise before color analysis
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # Convert BGR → HSV (HSV separates color from brightness, better for detection)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

    total_pixels = hsv.shape[0] * hsv.shape[1]  # 300 * 300 = 90,000

    # ── Step 2: Score Each Fruit ───────────────────────────────────────────
    scores = {}

    for fruit_name, profile in FRUIT_COLOR_PROFILES.items():
        # Combine all color range masks for this fruit
        combined_mask = np.zeros((300, 300), dtype=np.uint8)

        for lower, upper in profile['ranges']:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up the mask with morphological operations
        # This removes small noise pixels and fills gaps
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Count matching pixels
        matched_pixels = cv2.countNonZero(combined_mask)
        coverage = matched_pixels / total_pixels

        scores[fruit_name] = coverage

    # ── Step 3: Determine Winner ───────────────────────────────────────────
    # Pick the fruit with highest color coverage score
    best_fruit = max(scores, key=scores.get)
    best_score = scores[best_fruit]

    # ── Step 4: Confidence Calculation ────────────────────────────────────
    # Normalize confidence: compare best score vs sum of all scores
    total_score = sum(scores.values())

    if total_score > 0:
        confidence = (best_score / total_score) * 100
    else:
        confidence = 0.0

    # If confidence is too low, we're uncertain
    if best_score < 0.05:
        best_fruit = 'Unknown'
        confidence = 0.0

    # Round confidence percentage
    confidence = round(min(confidence, 99.0), 1)

    return {
        'fruit_type': best_fruit,
        'confidence': confidence,
        'color_scores': {k: round(v * 100, 2) for k, v in scores.items()},
        'dominant_score': round(best_score * 100, 2)
    }


def get_fruit_shape_features(image: np.ndarray) -> dict:
    """
    Extract basic shape features as secondary classification support.
    Uses contour analysis on the fruit region.

    Returns basic shape metrics (aspect ratio, roundness).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold to separate fruit from background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {'aspect_ratio': 1.0, 'roundness': 0.5}

    # Take the largest contour (assumed to be the fruit)
    largest = max(contours, key=cv2.contourArea)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = round(w / h if h > 0 else 1.0, 3)

    # Roundness: 4π·Area / Perimeter²  (1.0 = perfect circle)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    if perimeter > 0:
        roundness = round((4 * np.pi * area) / (perimeter ** 2), 3)
    else:
        roundness = 0.5

    return {
        'aspect_ratio': aspect_ratio,
        'roundness': roundness
    }
