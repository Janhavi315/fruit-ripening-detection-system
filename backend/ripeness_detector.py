"""
ripeness_detector.py - Fruit Ripeness Analysis Module
Calculates ripeness percentage using HSV color pixel analysis.
Each fruit has specific "ripe color zones" defined in HSV space.

Traditional CV Techniques Used:
- HSV color segmentation
- Pixel counting in color ranges
- Histogram analysis
- Morphological operations for noise removal
"""

import cv2
import numpy as np
import os


# ─── Ripeness Color Zones per Fruit ───────────────────────────────────────────
# These HSV ranges define what "ripe" looks like for each fruit.
# "Semi-ripe" gets partial credit. "Ripe" gets full credit.

RIPENESS_PROFILES = {
    'Mango': {
        'ripe': [
            # Deep yellow (fully ripe mango)
            (np.array([18, 120, 120]), np.array([35, 255, 255])),
            # Orange-yellow (peak ripeness)
            (np.array([10, 100, 120]), np.array([18, 255, 255])),
        ],
        'semi_ripe': [
            # Light yellow-green
            (np.array([35, 60, 100]), np.array([55, 200, 240])),
        ],
        'fruit_hue': [
            # Any mango-colored pixel (for total fruit area calculation)
            (np.array([10, 40, 80]), np.array([80, 255, 255])),
        ]
    },
    'Banana': {
        'ripe': [
            # Bright yellow (perfect ripe banana)
            (np.array([20, 120, 150]), np.array([32, 255, 255])),
        ],
        'semi_ripe': [
            # Yellow-green transition
            (np.array([32, 80, 120]), np.array([55, 220, 255])),
        ],
        'fruit_hue': [
            (np.array([15, 60, 100]), np.array([65, 255, 255])),
        ]
    },
    'Apple': {
        'ripe': [
            # Deep red (ripe apple) — two ranges because red wraps in HSV
            (np.array([0, 100, 80]), np.array([8, 255, 255])),
            (np.array([165, 100, 80]), np.array([179, 255, 255])),
        ],
        'semi_ripe': [
            # Pink-red (semi-ripe apple)
            (np.array([0, 60, 100]), np.array([15, 200, 255])),
            (np.array([155, 60, 100]), np.array([165, 200, 255])),
        ],
        'fruit_hue': [
            (np.array([0, 40, 60]), np.array([179, 255, 255])),
        ]
    },
    'Unknown': {
        'ripe': [
            (np.array([15, 80, 100]), np.array([40, 255, 255])),
        ],
        'semi_ripe': [
            (np.array([40, 40, 80]), np.array([80, 200, 200])),
        ],
        'fruit_hue': [
            (np.array([0, 30, 50]), np.array([179, 255, 255])),
        ]
    }
}

# Scoring weights: ripe pixels worth more than semi-ripe
RIPE_WEIGHT = 1.0
SEMI_RIPE_WEIGHT = 0.5

# Optional learned model (trained from data/ripe, data/unripe, data/rotten)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'ripeness_model.npz'
)
_CLASS_TO_PROFILE = {
    'unripe': {'status': 'Raw', 'target_pct': 18.0},
    'ripe': {'status': 'Fully Ripe', 'target_pct': 82.0},
    'rotten': {'status': 'Overripe', 'target_pct': 97.0},
}
_RIPENESS_MODEL = None


def _load_ripeness_model():
    """Load calibrated ripeness model from disk, if available."""
    global _RIPENESS_MODEL
    if _RIPENESS_MODEL is not None:
        return _RIPENESS_MODEL

    if not os.path.exists(_MODEL_PATH):
        _RIPENESS_MODEL = {}
        return _RIPENESS_MODEL

    try:
        model = np.load(_MODEL_PATH, allow_pickle=True)
        classes = [str(c) for c in model['classes'].tolist()]
        prototypes = model['prototypes'].astype(np.float32)
        means = model['class_mean_pct'].astype(np.float32)

        # Ensure rows are unit vectors for cosine similarity.
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        prototypes = prototypes / np.maximum(norms, 1e-8)

        _RIPENESS_MODEL = {
            'classes': classes,
            'prototypes': prototypes,
            'class_mean_pct': means
        }
    except Exception:
        _RIPENESS_MODEL = {}

    return _RIPENESS_MODEL


def _extract_hs_feature(image: np.ndarray) -> np.ndarray:
    """Extract normalized HS histogram feature for ripeness classification."""
    resized = cv2.resize(image, (256, 256))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = cv2.inRange(sat, 30, 255)
    val_mask = cv2.inRange(val, 35, 245)
    mask = cv2.bitwise_and(mask, val_mask)

    hist = cv2.calcHist([hsv], [0, 1], mask, [24, 24], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist


def _predict_from_calibrated_model(image: np.ndarray) -> dict:
    """Predict ripeness class probabilities from learned prototypes."""
    model = _load_ripeness_model()
    if not model:
        return {'available': False}

    feature = _extract_hs_feature(image)
    prototypes = model['prototypes']
    similarities = prototypes @ feature
    similarities = np.clip(similarities, -1.0, 1.0)
    probs = np.exp(similarities * 8.0)
    probs /= np.sum(probs)

    classes = model['classes']
    class_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}
    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    confidence = float(probs[top_idx] * 100.0)

    # Use class mean ripeness if available, else fallback to class mapping.
    class_means = model.get('class_mean_pct', np.zeros(len(classes), dtype=np.float32))
    weighted_pct = float(np.sum(probs * class_means))
    if weighted_pct <= 0:
        weighted_pct = _CLASS_TO_PROFILE.get(top_class, {}).get('target_pct', 50.0)

    return {
        'available': True,
        'class': top_class,
        'confidence': round(confidence, 1),
        'class_probs': class_probs,
        'estimated_pct': weighted_pct
    }


def calculate_ripeness(image: np.ndarray, fruit_type: str) -> dict:
    """
    Calculate ripeness percentage of the fruit using color pixel analysis.

    Algorithm:
    1. Convert to HSV
    2. Count total fruit-colored pixels (denominator)
    3. Count ripe + semi-ripe colored pixels (numerator, weighted)
    4. Ripeness % = weighted ripe pixels / total fruit pixels * 100
    5. Map percentage to status label and time prediction

    Args:
        image: BGR image (from cv2.imread)
        fruit_type: Detected fruit type (Mango/Banana/Apple/Unknown)

    Returns:
        dict with ripeness_percentage, status, days_to_ripe, color_distribution
    """

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    img = cv2.resize(image, (400, 400))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    calibrated_pred = _predict_from_calibrated_model(img)

    total_pixels = img.shape[0] * img.shape[1]  # 160,000

    # Get the color profile for this fruit
    profile = RIPENESS_PROFILES.get(fruit_type, RIPENESS_PROFILES['Unknown'])

    # ── Step 2: Calculate Fruit Area (background exclusion) ────────────────
    # Only count pixels that look like the fruit (exclude white/black background)
    fruit_mask = np.zeros((400, 400), dtype=np.uint8)
    for lower, upper in profile['fruit_hue']:
        mask = cv2.inRange(hsv, lower, upper)
        fruit_mask = cv2.bitwise_or(fruit_mask, mask)

    # Also exclude very dark and very bright (background) regions
    # Value channel: exclude near-black and near-white
    value_channel = hsv[:, :, 2]
    bright_mask = cv2.inRange(value_channel, np.array([40]), np.array([240]))
    sat_channel = hsv[:, :, 1]
    sat_mask = cv2.inRange(sat_channel, np.array([20]), np.array([255]))

    # Combine: must be fruit-colored AND have reasonable brightness/saturation
    fruit_area_mask = cv2.bitwise_and(fruit_mask, bright_mask)
    fruit_area_mask = cv2.bitwise_and(fruit_area_mask, sat_mask)

    # Clean mask with morphological ops
    kernel = np.ones((7, 7), np.uint8)
    fruit_area_mask = cv2.morphologyEx(fruit_area_mask, cv2.MORPH_CLOSE, kernel)
    fruit_area_mask = cv2.morphologyEx(fruit_area_mask, cv2.MORPH_OPEN, kernel)

    fruit_pixel_count = cv2.countNonZero(fruit_area_mask)

    # Fallback: if fruit area is too small, use total pixels
    if fruit_pixel_count < total_pixels * 0.05:
        fruit_pixel_count = int(total_pixels * 0.3)

    # ── Step 3: Count Ripe Pixels ──────────────────────────────────────────
    ripe_mask = np.zeros((400, 400), dtype=np.uint8)
    for lower, upper in profile['ripe']:
        mask = cv2.inRange(hsv, lower, upper)
        ripe_mask = cv2.bitwise_or(ripe_mask, mask)

    # Only count ripe pixels within the fruit area
    ripe_mask = cv2.bitwise_and(ripe_mask, fruit_area_mask)
    ripe_pixels = cv2.countNonZero(ripe_mask)

    # ── Step 4: Count Semi-Ripe Pixels ────────────────────────────────────
    semi_mask = np.zeros((400, 400), dtype=np.uint8)
    for lower, upper in profile['semi_ripe']:
        mask = cv2.inRange(hsv, lower, upper)
        semi_mask = cv2.bitwise_or(semi_mask, mask)

    semi_mask = cv2.bitwise_and(semi_mask, fruit_area_mask)
    semi_pixels = cv2.countNonZero(semi_mask)

    # ── Step 5: Weighted Ripeness Score ───────────────────────────────────
    weighted_ripe = (ripe_pixels * RIPE_WEIGHT) + (semi_pixels * SEMI_RIPE_WEIGHT)
    ripeness_pct = (weighted_ripe / fruit_pixel_count) * 100

    # Clamp between 0 and 100
    ripeness_pct = max(0.0, min(100.0, ripeness_pct))

    # ── Step 6: HSV Histogram Analysis for fine-tuning ────────────────────
    # Analyze hue histogram to refine our estimate
    hue_hist = cv2.calcHist([hsv], [0], fruit_area_mask, [180], [0, 180])
    hue_hist = hue_hist.flatten()
    dominant_hue = int(np.argmax(hue_hist))

    # Hue-based adjustment (fine-tuning)
    ripeness_pct = _adjust_ripeness_by_hue(ripeness_pct, dominant_hue, fruit_type)
    ripeness_pct = round(ripeness_pct, 1)

    # ── Step 7: Fuse data-driven and rule-based ripeness ───────────────────
    status = _get_ripeness_status(ripeness_pct)
    if calibrated_pred.get('available'):
        model_pct = calibrated_pred['estimated_pct']
        ripeness_pct = round((0.35 * ripeness_pct) + (0.65 * model_pct), 1)

        predicted_class = calibrated_pred.get('class', '')
        predicted_status = _CLASS_TO_PROFILE.get(predicted_class, {}).get('status')
        if predicted_status and calibrated_pred.get('confidence', 0) >= 45:
            status = predicted_status
        else:
            status = _get_ripeness_status(ripeness_pct)

    # ── Step 8: Status and Time Prediction ────────────────────────────────
    days_to_ripe = _predict_days_to_ripe(ripeness_pct)

    # ── Step 9: Color Distribution for Charts ─────────────────────────────
    if calibrated_pred.get('available'):
        class_probs = calibrated_pred['class_probs']
        raw_pct = class_probs.get('unripe', 0.0) * 100.0
        semi_pct = max(0.0, 100.0 - raw_pct - (class_probs.get('ripe', 0.0) * 100.0))
        ripe_pct = class_probs.get('ripe', 0.0) * 100.0
        # Treat rotten confidence as overripe component.
        rotten_component = class_probs.get('rotten', 0.0) * 100.0
        ripe_pct = min(100.0, ripe_pct + (rotten_component * 0.5))
        raw_pct = max(0.0, raw_pct - (rotten_component * 0.2))
    else:
        raw_pct = max(0, 100 - ripeness_pct - (semi_pixels / fruit_pixel_count * 100 * 0.5))
        semi_pct = min(semi_pixels / fruit_pixel_count * 100, 100 - ripeness_pct)
        ripe_pct = ripeness_pct

    # Normalize to 100%
    total = raw_pct + semi_pct + ripe_pct
    if total > 0:
        raw_pct = round(raw_pct / total * 100, 1)
        semi_pct = round(semi_pct / total * 100, 1)
        ripe_pct = round(ripe_pct / total * 100, 1)

    return {
        'ripeness_percentage': ripeness_pct,
        'ripeness_status': status,
        'days_to_ripe': days_to_ripe,
        'dominant_hue': dominant_hue,
        'fruit_pixel_count': fruit_pixel_count,
        'ripe_pixels': ripe_pixels,
        'semi_pixels': semi_pixels,
        'model_confidence': calibrated_pred.get('confidence', 0.0),
        'model_class': calibrated_pred.get('class', 'unavailable'),
        'color_distribution': {
            'raw': raw_pct,
            'semi_ripe': semi_pct,
            'ripe': ripe_pct
        },
        'growth_timeline': _generate_growth_timeline(ripeness_pct, days_to_ripe)
    }


def _adjust_ripeness_by_hue(ripeness_pct: float, dominant_hue: int, fruit_type: str) -> float:
    """
    Fine-tune ripeness based on dominant hue value.
    This adds accuracy beyond just pixel counting.
    """
    if fruit_type == 'Mango':
        # Hue 15-25 = yellow (ripe), Hue 35-65 = green (raw)
        if 15 <= dominant_hue <= 25:
            ripeness_pct = min(100, ripeness_pct * 1.15)
        elif 35 <= dominant_hue <= 65:
            ripeness_pct = ripeness_pct * 0.7

    elif fruit_type == 'Banana':
        # Hue 22-30 = yellow (ripe), Hue 35-55 = green (raw)
        if 22 <= dominant_hue <= 30:
            ripeness_pct = min(100, ripeness_pct * 1.2)
        elif 35 <= dominant_hue <= 55:
            ripeness_pct = ripeness_pct * 0.65

    elif fruit_type == 'Apple':
        # Hue 0-10 or 165-179 = red (ripe)
        if dominant_hue <= 10 or dominant_hue >= 165:
            ripeness_pct = min(100, ripeness_pct * 1.1)

    return ripeness_pct


def _get_ripeness_status(ripeness_pct: float) -> str:
    """Map ripeness percentage to human-readable status."""
    if ripeness_pct <= 30:
        return 'Raw'
    elif ripeness_pct <= 70:
        return 'Semi Ripe'
    elif ripeness_pct <= 90:
        return 'Fully Ripe'
    else:
        return 'Overripe'


def _predict_days_to_ripe(ripeness_pct: float) -> float:
    """
    Estimate days until fruit reaches full ripeness.
    Formula: days = (100 - ripeness%) / 12
    At ~12% ripeness gain per day under normal conditions.
    """
    if ripeness_pct >= 90:
        return 0.0
    days = (100 - ripeness_pct) / 12
    return round(days, 1)


def _generate_growth_timeline(current_pct: float, days_to_ripe: float) -> list:
    """
    Generate projected ripeness timeline data points for the chart.
    Returns a list of {day, ripeness} objects.
    """
    timeline = []

    # Go back 3 days to show past growth
    start_pct = max(0, current_pct - 3 * 12)

    # Total span: 3 days past + current + future until 100%
    total_days = int(days_to_ripe) + 4

    for day_offset in range(-3, total_days + 1):
        projected_pct = current_pct + (day_offset * 12)
        projected_pct = max(0, min(100, projected_pct))
        timeline.append({
            'day': day_offset,
            'ripeness': round(projected_pct, 1),
            'is_current': day_offset == 0
        })

    return timeline
