"""
app.py - Flask REST API Entry Point
Smart Fruit Ripeness and Quality Detection System

Routes:
  POST /api/analyze    - Main analysis endpoint (upload + process image)
  GET  /api/history    - Fetch recent scan history
  GET  /api/health     - API health check
  GET  /               - Serve frontend index.html
"""

import os
import sys
import uuid
import json
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ── Add backend directory to Python path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fruit_classifier import classify_fruit, get_fruit_shape_features
from ripeness_detector import calculate_ripeness
from disease_detector import detect_disease, get_disease_recommendations
from database import init_db, get_fruit_knowledge, save_scan_result, get_recent_scans
from utils.image_utils import (
    load_image, validate_image, image_to_base64,
    resize_for_display, preprocess_for_analysis, get_image_stats, allowed_file
)

# ── Flask App Configuration ────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'),
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
)



# Upload folder configuration
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'uploads'
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
# ── Manual CORS Headers (no flask-cors needed) ─────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ── Initialize Database on Startup ────────────────────────────────────────────
init_db()
print("[APP] Fruit Ripeness Detection System started!")
print(f"[APP] Upload folder: {UPLOAD_FOLDER}")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Serve Frontend
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main frontend HTML page."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'
    )
    return send_from_directory(frontend_path, 'index.html')


@app.route('/result')
def result():
    """Serve the result dashboard HTML page."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'
    )
    return send_from_directory(frontend_path, 'result.html')


@app.route('/frontend/<path:filename>')
def frontend_files(filename):
    """Serve frontend static files (CSS, JS)."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'
    )
    return send_from_directory(frontend_path, filename)


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve uploaded images and static assets."""
    static_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'static'
    )
    return send_from_directory(static_path, filename)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check to confirm API is running."""
    return jsonify({
        'status': 'healthy',
        'message': 'Fruit Ripeness Detection API is running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Main Analysis Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def analyze_fruit():
    """
    Main fruit analysis endpoint.

    Request: multipart/form-data with 'image' file field
    Response: JSON with complete analysis results

    Pipeline:
    1. Validate and save uploaded image
    2. Load and preprocess image
    3. Classify fruit type
    4. Calculate ripeness
    5. Detect disease
    6. Fetch storage/market knowledge
    7. Generate alerts/notifications
    8. Save to database
    9. Return complete result JSON
    """

    # ── Step 1: Validate Upload ────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Please upload an image.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, WEBP, BMP'
        }), 400

    # ── Step 2: Save Uploaded File ─────────────────────────────────────────
    # Generate unique filename to avoid collisions
    original_name = secure_filename(file.filename)
    ext = original_name.rsplit('.', 1)[1].lower() if '.' in original_name else 'jpg'
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # ── Step 3: Validate Image ─────────────────────────────────────────
        is_valid, error_msg = validate_image(filepath)
        if not is_valid:
            os.remove(filepath)  # Clean up invalid file
            return jsonify({'error': error_msg}), 400

        # ── Step 4: Load and Preprocess Image ─────────────────────────────
        raw_image = load_image(filepath)
        processed_image = preprocess_for_analysis(raw_image)
        img_stats = get_image_stats(raw_image)

        print(f"[ANALYZE] Processing: {unique_filename} | Size: {raw_image.shape}")

        # ── Step 5: Classify Fruit Type ────────────────────────────────────
        print("[ANALYZE] Step 1/4: Classifying fruit type...")
        classification = classify_fruit(processed_image)
        fruit_type = classification['fruit_type']
        shape_features = get_fruit_shape_features(processed_image)

        print(f"[ANALYZE]   → Detected: {fruit_type} (confidence: {classification['confidence']}%)")

        # ── Step 6: Calculate Ripeness ─────────────────────────────────────
        print("[ANALYZE] Step 2/4: Calculating ripeness...")
        ripeness_data = calculate_ripeness(processed_image, fruit_type)

        print(f"[ANALYZE]   → Ripeness: {ripeness_data['ripeness_percentage']}% ({ripeness_data['ripeness_status']})")

        # ── Step 7: Detect Disease ─────────────────────────────────────────
        print("[ANALYZE] Step 3/4: Detecting disease/damage...")
        disease_data = detect_disease(processed_image, fruit_type)

        # Get the annotated image (with red circles on spots)
        annotated_img = disease_data.pop('annotated_image', processed_image)
        annotated_b64 = image_to_base64(resize_for_display(annotated_img, 500))

        # Also encode original image for display
        original_b64 = image_to_base64(resize_for_display(raw_image, 500))

        print(f"[ANALYZE]   → Disease: {'YES' if disease_data['disease_detected'] else 'No'} | Area: {disease_data['disease_area_pct']}%")

        # ── Step 8: Get Storage & Market Knowledge ─────────────────────────
        print("[ANALYZE] Step 4/4: Fetching recommendations...")
        knowledge = get_fruit_knowledge(fruit_type)
        disease_recs = get_disease_recommendations(
            disease_data['severity'], fruit_type
        )

        # Select appropriate storage and market advice based on ripeness status
        status = ripeness_data['ripeness_status']
        status_key = status.lower().replace(' ', '_')

        storage_tip = knowledge.get(f'storage_{status_key}', knowledge.get('storage_ripe', ''))
        market_tip = knowledge.get(f'market_{status_key}', knowledge.get('market_ripe', ''))

        # Handle 'overripe' case (maps to 'ripe' storage but with warning)
        if status == 'Overripe':
            storage_tip = knowledge.get('storage_ripe', '') + ' ⚠️ Consume immediately!'
            market_tip = 'Not recommended for fresh market. Use for juice or processing.'

        # ── Step 9: Generate Alerts / Notifications ────────────────────────
        alerts = _generate_alerts(
            ripeness_data['ripeness_status'],
            ripeness_data['ripeness_percentage'],
            disease_data['disease_detected'],
            disease_data['severity'],
            disease_data['spots_count']
        )

        # ── Step 10: Save to Database ──────────────────────────────────────
        scan_id = save_scan_result(
            filename=unique_filename,
            fruit_type=fruit_type,
            ripeness_pct=ripeness_data['ripeness_percentage'],
            ripeness_status=ripeness_data['ripeness_status'],
            days_to_ripe=ripeness_data['days_to_ripe'],
            disease_detected=disease_data['disease_detected'],
            disease_area_pct=disease_data['disease_area_pct']
        )

        # ── Step 11: Build Complete Response ──────────────────────────────
        response = {
            'success': True,
            'scan_id': scan_id,
            'timestamp': datetime.now().isoformat(),

            # ── Images ──
            'images': {
                'original': original_b64,
                'annotated': annotated_b64,
                'filename': unique_filename
            },

            # ── Fruit Classification ──
            'fruit': {
                'type': fruit_type,
                'confidence': classification['confidence'],
                'color_scores': classification['color_scores'],
                'shape': shape_features,
                'image_stats': img_stats
            },

            # ── Ripeness Analysis ──
            'ripeness': {
                'percentage': ripeness_data['ripeness_percentage'],
                'status': ripeness_data['ripeness_status'],
                'days_to_ripe': ripeness_data['days_to_ripe'],
                'color_distribution': ripeness_data['color_distribution'],
                'growth_timeline': ripeness_data['growth_timeline']
            },

            # ── Disease Analysis ──
            'disease': {
                'detected': disease_data['disease_detected'],
                'area_percentage': disease_data['disease_area_pct'],
                'spots_count': disease_data['spots_count'],
                'severity': disease_data['severity'],
                'recommendation': disease_recs
            },

            # ── Recommendations ──
            'recommendations': {
                'storage': storage_tip,
                'market_stage': market_tip,
                'ideal_temp': knowledge.get('ideal_temp_c', 'N/A'),
                'shelf_life': knowledge.get('shelf_life_days', 'N/A'),
                'fun_fact': knowledge.get('fun_fact', ''),
                'disease_action': disease_recs
            },

            # ── Alerts ──
            'alerts': alerts
        }

        print(f"[ANALYZE] ✓ Complete. Scan ID: {scan_id}")
        return jsonify(response), 200

    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(filepath):
            os.remove(filepath)

        error_detail = traceback.format_exc()
        print(f"[ERROR] Analysis failed: {str(e)}\n{error_detail}")

        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'detail': 'Please ensure the uploaded image shows a clear view of a fruit.'
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: Scan History
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
def get_history():
    """Return the last N scan results from database."""
    limit = request.args.get('limit', 10, type=int)
    scans = get_recent_scans(limit)
    return jsonify({
        'success': True,
        'count': len(scans),
        'scans': scans
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Generate Alert Notifications
# ─────────────────────────────────────────────────────────────────────────────

def _generate_alerts(status: str, ripeness_pct: float,
                     disease_detected: bool, severity: str, spots_count: int) -> list:
    """
    Generate alert notifications based on analysis results.

    Alert types:
    - overripe: Fruit has passed peak quality
    - disease: Disease or significant damage detected
    - ready_to_sell: Fruit is at optimal market stage
    - warning: Semi-ripe approaching ready stage
    - info: General information
    """
    alerts = []

    # ── Overripe Alert ─────────────────────────────────────────────────────
    if status == 'Overripe' or ripeness_pct > 90:
        alerts.append({
            'type': 'overripe',
            'level': 'danger',
            'icon': '⚠️',
            'title': 'Overripe Alert!',
            'message': f'This fruit is overripe ({ripeness_pct:.0f}% ripeness). '
                      f'Sell or consume immediately. Quality deteriorating rapidly.',
            'action': 'Use for juice, smoothies, or processing immediately.'
        })

    # ── Disease Alert ──────────────────────────────────────────────────────
    if disease_detected:
        if severity == 'Severe':
            alerts.append({
                'type': 'disease',
                'level': 'danger',
                'icon': '🦠',
                'title': 'Severe Disease Detected!',
                'message': f'Significant disease/damage found ({spots_count} spots). '
                          f'Do NOT send to market.',
                'action': 'Remove from storage. Do not mix with healthy fruits.'
            })
        elif severity == 'Moderate':
            alerts.append({
                'type': 'disease',
                'level': 'warning',
                'icon': '⚠️',
                'title': 'Disease Warning',
                'message': f'Moderate damage detected ({spots_count} spots). '
                          f'Quality affected. Local market only.',
                'action': 'Separate from batch. Monitor for 24 hours.'
            })
        elif severity == 'Mild':
            alerts.append({
                'type': 'disease',
                'level': 'info',
                'icon': 'ℹ️',
                'title': 'Minor Spots Detected',
                'message': f'Small blemishes detected ({spots_count} spots). '
                          f'Acceptable quality for local market.',
                'action': 'Monitor closely. Consume within expected timeline.'
            })

    # ── Ready to Sell Alert ────────────────────────────────────────────────
    if status == 'Fully Ripe' and not disease_detected:
        alerts.append({
            'type': 'ready',
            'level': 'success',
            'icon': '✅',
            'title': 'Ready to Sell!',
            'message': f'Fruit is at peak quality ({ripeness_pct:.0f}% ripeness). '
                      f'Excellent market condition!',
            'action': 'Send to market within 24-48 hours for best price.'
        })

    # ── Semi-Ripe Approaching Warning ─────────────────────────────────────
    elif status == 'Semi Ripe' and 55 <= ripeness_pct <= 70:
        alerts.append({
            'type': 'warning',
            'level': 'warning',
            'icon': '🕐',
            'title': 'Approaching Peak Ripeness',
            'message': f'Fruit is {ripeness_pct:.0f}% ripe. Will be market-ready soon.',
            'action': 'Prepare for market dispatch in 1-2 days.'
        })

    # ── Raw / Early Stage Info ─────────────────────────────────────────────
    elif status == 'Raw':
        alerts.append({
            'type': 'info',
            'level': 'info',
            'icon': '🌱',
            'title': 'Early Stage',
            'message': f'Fruit is still raw ({ripeness_pct:.0f}% ripe). '
                      f'Requires more time before market.',
            'action': 'Continue storage. Check again in 2-3 days.'
        })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum upload size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found.'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Smart Fruit Ripeness Detection System v1.0")
    print("  Starting Flask Development Server...")
    print("  URL: http://localhost:5000")
    print("=" * 60)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
