"""
disease_detector.py - Fruit Disease and Damage Detection Module
Detects dark patches, bruises, rot spots using HSV brightness analysis.

Traditional CV Techniques:
- HSV Value channel thresholding (dark = low Value)
- Contour detection for spot shape analysis
- Area percentage calculation
- Blob detection for spot counting
"""

import cv2
import numpy as np


# ─── Detection Thresholds ──────────────────────────────────────────────────────
DISEASE_AREA_THRESHOLD = 0.08      # 8% of fruit area = disease warning
BRUISE_DARK_THRESHOLD = 50         # HSV Value below 50 = potential bruise/disease spot
MIN_SPOT_AREA = 80                 # Minimum pixel area to count as a real spot (noise filter)


def detect_disease(image: np.ndarray, fruit_type: str = 'Unknown') -> dict:
    """
    Detect disease spots, bruises, and damage on fruit surface.

    Algorithm:
    1. Convert to HSV, isolate the fruit region
    2. Extract Value channel (brightness)
    3. Threshold dark regions → potential disease spots
    4. Filter by size (remove noise)
    5. Calculate disease area as % of total fruit area
    6. Return severity assessment

    Args:
        image: BGR image from cv2.imread
        fruit_type: For fruit-specific thresholds

    Returns:
        dict with disease_detected, disease_area_pct, spots_count, severity, annotated_image_base64
    """

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    img = cv2.resize(image, (400, 400))
    img_display = img.copy()  # Keep clean copy for annotation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── Step 2: Isolate Fruit Region (exclude pure background) ────────────
    # Fruit region = pixels with reasonable saturation (not white background)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Fruit pixels: moderate saturation + not too dark + not too bright
    fruit_region_mask = cv2.inRange(
        saturation,
        np.array([15]),   # Minimum saturation
        np.array([255])
    )
    # Also exclude near-white background
    not_white = cv2.inRange(value, np.array([30]), np.array([230]))
    fruit_region_mask = cv2.bitwise_and(fruit_region_mask, not_white)

    # Clean up
    kernel = np.ones((9, 9), np.uint8)
    fruit_region_mask = cv2.morphologyEx(fruit_region_mask, cv2.MORPH_CLOSE, kernel)
    fruit_region_mask = cv2.morphologyEx(fruit_region_mask, cv2.MORPH_OPEN, kernel)

    fruit_pixel_count = max(cv2.countNonZero(fruit_region_mask), 1000)

    # ── Step 3: Detect Dark Patches (Disease/Bruise Zones) ────────────────
    # Low Value = dark = potential disease, bruise, or rot
    dark_mask = cv2.inRange(value, np.array([0]), np.array([BRUISE_DARK_THRESHOLD]))

    # Only consider dark patches ON the fruit (not background)
    dark_on_fruit = cv2.bitwise_and(dark_mask, fruit_region_mask)

    # ── Step 4: Also detect abnormal color patches ─────────────────────────
    # Brown/dark-brown spots (common in over-ripe or diseased fruit)
    # In HSV: brownish = Hue 10-20, low-mid saturation, low-mid value
    brown_lower = np.array([5, 40, 20])
    brown_upper = np.array([20, 180, 80])
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    brown_on_fruit = cv2.bitwise_and(brown_mask, fruit_region_mask)

    # Combine dark + brown spots
    disease_mask = cv2.bitwise_or(dark_on_fruit, brown_on_fruit)

    # ── Step 5: Morphological Cleanup ─────────────────────────────────────
    spot_kernel = np.ones((5, 5), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, spot_kernel)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, spot_kernel)

    # ── Step 6: Find and Filter Contours (spots) ──────────────────────────
    contours, _ = cv2.findContours(
        disease_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter: only spots larger than MIN_SPOT_AREA pixels
    valid_spots = [c for c in contours if cv2.contourArea(c) > MIN_SPOT_AREA]
    spots_count = len(valid_spots)

    # ── Step 7: Calculate Disease Area Percentage ─────────────────────────
    disease_pixel_count = sum(cv2.contourArea(c) for c in valid_spots)
    disease_area_pct = (disease_pixel_count / fruit_pixel_count) * 100
    disease_area_pct = round(min(disease_area_pct, 100.0), 2)

    # ── Step 8: Disease Detection Decision ────────────────────────────────
    disease_detected = disease_area_pct > (DISEASE_AREA_THRESHOLD * 100)

    # ── Step 9: Severity Classification ───────────────────────────────────
    severity = _classify_severity(disease_area_pct, spots_count)

    # ── Step 10: Draw Annotations on Display Image ────────────────────────
    for contour in valid_spots:
        # Draw red outline around each detected spot
        cv2.drawContours(img_display, [contour], -1, (0, 0, 255), 2)
        # Get bounding box and add label
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 0, 220), 1)

    # Overlay fruit region mask (semi-transparent green tint for healthy area)
    # This helps visualize which region was analyzed
    overlay = img_display.copy()
    overlay[fruit_region_mask > 0] = overlay[fruit_region_mask > 0] * [1.0, 1.1, 0.9]
    img_display = cv2.addWeighted(img_display, 0.8, overlay, 0.2, 0)

    # ── Step 11: Spot Details ──────────────────────────────────────────────
    spot_details = []
    for i, c in enumerate(valid_spots[:10]):  # Max 10 spots in response
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        spot_details.append({
            'spot_id': i + 1,
            'area_pixels': int(area),
            'center_x': cx,
            'center_y': cy
        })

    return {
        'disease_detected': disease_detected,
        'disease_area_pct': disease_area_pct,
        'spots_count': spots_count,
        'severity': severity,
        'fruit_area_pixels': fruit_pixel_count,
        'disease_pixels': int(disease_pixel_count),
        'spot_details': spot_details,
        'annotated_image': img_display   # Will be encoded to base64 in app.py
    }


def _classify_severity(area_pct: float, spots_count: int) -> str:
    """
    Classify disease severity based on area percentage and spot count.

    Severity Levels:
    - None: No disease detected
    - Mild: Small isolated spots, early warning
    - Moderate: Multiple spots or medium area affected
    - Severe: Large area affected or many spots
    """
    if area_pct <= 0.5 and spots_count == 0:
        return 'None'
    elif area_pct <= 5.0 and spots_count <= 2:
        return 'Mild'
    elif area_pct <= 15.0 or spots_count <= 6:
        return 'Moderate'
    else:
        return 'Severe'


def get_disease_recommendations(severity: str, fruit_type: str) -> dict:
    """
    Return action recommendations based on disease severity.
    """
    recommendations = {
        'None': {
            'action': 'No action needed',
            'description': 'Fruit appears healthy with no visible disease or damage.',
            'sellable': True,
            'storage_modifier': 'Normal storage applies.'
        },
        'Mild': {
            'action': 'Monitor closely',
            'description': 'Minor spots detected. Monitor daily and consume/sell soon.',
            'sellable': True,
            'storage_modifier': 'Consume within 1-2 days sooner than usual.'
        },
        'Moderate': {
            'action': 'Sort and separate',
            'description': 'Moderate spotting detected. Separate from healthy fruits to prevent spread.',
            'sellable': False,
            'storage_modifier': 'Do not store with other fruits. Use immediately.'
        },
        'Severe': {
            'action': 'Do NOT sell — immediate action required',
            'description': 'Severe disease or damage detected. Fruit is not suitable for market.',
            'sellable': False,
            'storage_modifier': 'Discard or use only for processing/compost.'
        }
    }

    return recommendations.get(severity, recommendations['None'])
