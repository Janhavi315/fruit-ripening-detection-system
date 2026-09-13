"""
database.py - SQLite Database Module
Handles all database operations for the Fruit Ripeness Detection System
"""

import sqlite3
import os

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database.db')


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Allows dict-like access to rows
    return conn


def init_db():
    """
    Initialize the database with all required tables and seed data.
    Called once when the application starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ─── Table: scan_history ───────────────────────────────────────────────
    # Stores every fruit scan performed by users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL,
            fruit_type      TEXT,
            ripeness_pct    REAL,
            ripeness_status TEXT,
            days_to_ripe    REAL,
            disease_detected INTEGER DEFAULT 0,
            disease_area_pct REAL DEFAULT 0.0,
            scan_timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ─── Table: fruit_knowledge ────────────────────────────────────────────
    # Static knowledge base: storage tips, market advice, ripeness thresholds
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fruit_knowledge (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fruit_type      TEXT UNIQUE NOT NULL,
            storage_raw     TEXT,
            storage_semi    TEXT,
            storage_ripe    TEXT,
            market_raw      TEXT,
            market_semi     TEXT,
            market_ripe     TEXT,
            ideal_temp_c    TEXT,
            shelf_life_days TEXT,
            fun_fact        TEXT
        )
    ''')

    # ─── Seed fruit knowledge data ─────────────────────────────────────────
    fruits = [
        (
            'Mango',
            'Store at room temperature (20–25°C). Do NOT refrigerate raw mangoes — cold halts ripening.',
            'Keep at room temperature. Check daily. Separate from other fruits to slow ripening.',
            'Refrigerate at 8–12°C once ripe. Consume within 3–5 days. Keep away from direct sunlight.',
            'Not recommended. Raw mangoes are too astringent for fresh market. Use for pickles or chutneys.',
            'Ideal for export and wholesale markets. Shelf life during transit is excellent at this stage.',
            'Perfect stage for local fresh markets and retail. Best eating quality and aroma.',
            '8–12°C (ripe), 20–25°C (raw)',
            '5–10 days (raw to ripe), 3–5 days (ripe in fridge)',
            'Mangoes release ethylene gas which speeds up ripening. Store with bananas to ripen faster!'
        ),
        (
            'Banana',
            'Keep at room temperature. Hang them if possible — reduces bruising and slows browning.',
            'Room temperature is best. Wrap stem tips with plastic wrap to slow ripening by 3–5 days.',
            'Refrigerate once fully ripe. Skin turns black but inside stays fresh for 2–3 extra days.',
            'Not suitable for fresh market. Use for green banana chips or export to distant markets.',
            'Excellent for retail and supermarkets. Can ripen in 1–2 days in warm conditions.',
            'Best for local market immediate sale. Baking, smoothies, or direct consumption.',
            '13–15°C (raw), 18–20°C (ripening)',
            '7–14 days (raw to ripe), 2–5 days once ripe',
            'Never refrigerate green bananas — below 13°C causes chilling injury and prevents ripening!'
        ),
        (
            'Apple',
            'Refrigerate immediately. Apples last 6–8 weeks in the fridge. Keep away from vegetables.',
            'Refrigerate. Consume within 2–3 weeks. Good crunch and nutrition at this stage.',
            'Can be stored at room temp for 1–2 weeks. Refrigerate to extend up to 4 weeks.',
            'Best for export and cold-chain markets. Long shelf life advantage.',
            'Good for supermarkets and retail with proper cold storage.',
            'Ready for all markets. Fresh eating quality at its peak.',
            '0–4°C (best), up to 10°C (acceptable)',
            '4–8 weeks (refrigerated), 1–2 weeks (room temp)',
            'Apples produce the most ethylene gas of any common fruit — store separately from other produce!'
        )
    ]

    cursor.executemany('''
        INSERT OR IGNORE INTO fruit_knowledge
        (fruit_type, storage_raw, storage_semi, storage_ripe,
         market_raw, market_semi, market_ripe,
         ideal_temp_c, shelf_life_days, fun_fact)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', fruits)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


def get_fruit_knowledge(fruit_type: str) -> dict:
    """
    Fetch storage and market knowledge for a specific fruit.
    Returns a dictionary with all knowledge fields.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM fruit_knowledge WHERE fruit_type = ?', (fruit_type,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    # Fallback if fruit not in DB
    return {
        'fruit_type': fruit_type,
        'storage_raw': 'Store in a cool, dry place.',
        'storage_semi': 'Store at room temperature.',
        'storage_ripe': 'Refrigerate and consume soon.',
        'market_raw': 'Not ideal for fresh market.',
        'market_semi': 'Suitable for wholesale.',
        'market_ripe': 'Ready for retail market.',
        'ideal_temp_c': '10–20°C',
        'shelf_life_days': '3–7 days',
        'fun_fact': 'Fresh fruits are packed with vitamins!'
    }


def save_scan_result(filename, fruit_type, ripeness_pct, ripeness_status,
                     days_to_ripe, disease_detected, disease_area_pct) -> int:
    """
    Save a scan result to the database.
    Returns the inserted row's ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO scan_history
        (filename, fruit_type, ripeness_pct, ripeness_status,
         days_to_ripe, disease_detected, disease_area_pct)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (filename, fruit_type, ripeness_pct, ripeness_status,
          days_to_ripe, int(disease_detected), disease_area_pct))
    conn.commit()
    scan_id = cursor.lastrowid
    conn.close()
    return scan_id


def get_recent_scans(limit: int = 10) -> list:
    """Retrieve the most recent scan results."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM scan_history
        ORDER BY scan_timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
