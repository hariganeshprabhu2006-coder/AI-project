from flask import Flask, request, redirect, url_for, render_template_string, flash, jsonify
import sqlite3
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from ultralytics import YOLO
import cv2

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent
DB = BASE_DIR / "products.db"
MODEL_PATH = BASE_DIR / "weights.pt"

# Your latest phone IP address:
CAMERA_URL = "http://10.31.22.96:8080/video"
AI_CONFIDENCE = 0.18
DETECT_IMGSZ = 640
DETECT_FRAMES = 2  # two recent frames reduce one-frame misses; increase only if speed allows
KNOWN_CLASSES = ["denver_spray", "parachute_oil", "pepsodent", "red_chana"]

app = Flask(__name__)
app.secret_key = "smart-shelf-local-key"

MODEL = None
MODEL_LOAD_ERROR = None
try:
    MODEL = YOLO(str(MODEL_PATH))
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)

# Use NVIDIA GPU automatically when PyTorch CUDA is available; otherwise CPU.
try:
    import torch
    INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
    if INFERENCE_DEVICE == "cpu":
        # Avoid excessive CPU thread contention on typical laptops.
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
except Exception:
    torch = None
    INFERENCE_DEVICE = "cpu"

try:
    from pyzbar.pyzbar import decode as decode_barcodes
    BARCODE_IMPORT_ERROR = None
except Exception as exc:
    decode_barcodes = None
    BARCODE_IMPORT_ERROR = str(exc)


# ==================== DATABASE ====================
def connect_db():
    return sqlite3.connect(str(DB), timeout=10)


def initialize_database():
    """Keep existing products; create the table or add min_stock only if needed."""
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                barcode TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL NOT NULL DEFAULT 0,
                expiry_date TEXT,
                quantity INTEGER NOT NULL DEFAULT 1
            )
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(products)").fetchall()}
        if "min_stock" not in columns:
            conn.execute("ALTER TABLE products ADD COLUMN min_stock INTEGER NOT NULL DEFAULT 2")
        if "discount_percent" not in columns:
            conn.execute("ALTER TABLE products ADD COLUMN discount_percent REAL NOT NULL DEFAULT 0")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                barcode TEXT NOT NULL,
                product_name TEXT NOT NULL,
                action TEXT NOT NULL,
                old_quantity INTEGER,
                new_quantity INTEGER,
                change_amount INTEGER NOT NULL DEFAULT 0,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
        """)


def parse_discount(value):
    """Accept both '10' and '10%' and validate a safe percentage."""
    text = (value or "0").strip().replace("%", "").strip()
    discount = float(text or 0)
    if not 0 <= discount <= 90:
        raise ValueError("Discount must be between 0 and 90 percent.")
    return discount


def log_stock_change(conn, barcode, product_name, action, old_quantity, new_quantity, note=""):
    """Insert history using the actual existing table schema, including legacy required columns."""
    info = conn.execute("PRAGMA table_info(stock_history)").fetchall()
    if not info:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    change_amount = int(new_quantity or 0) - int(old_quantity or 0)
    known = {
        "barcode": barcode, "product_name": product_name, "action": action,
        "old_quantity": old_quantity, "new_quantity": new_quantity,
        "change_amount": change_amount, "note": note,
        "created_at": now, "timestamp": now, "event_time": now,
        "event": action, "details": note, "description": note,
        "quantity_change": change_amount, "product_barcode": barcode,
    }
    fields, values = [], []
    for col in info:
        name, notnull, default_value, pk = col[1], col[3], col[4], col[5]
        if pk and "INT" in str(col[2]).upper():
            continue
        if name in known:
            fields.append(name); values.append(known[name])
        elif notnull and default_value is None:
            # Legacy tables may have extra mandatory columns. Give safe values based on type.
            typ = str(col[2]).upper()
            fields.append(name)
            if "INT" in typ or "REAL" in typ or "NUM" in typ:
                values.append(0)
            else:
                values.append(now if ("date" in name.lower() or "time" in name.lower()) else "")
    if fields:
        placeholders = ", ".join("?" for _ in fields)
        conn.execute(f"INSERT INTO stock_history ({', '.join(fields)}) VALUES ({placeholders})", values)


def status_for(expiry):
    if not expiry:
        return "", "No expiry date"
    try:
        expiry_day = date.fromisoformat(expiry)
    except (ValueError, TypeError):
        return "warn", "Check expiry date"
    if expiry_day < date.today():
        return "bad", "Expired"
    if expiry_day <= date.today() + timedelta(days=30):
        return "warn", "Near expiry"
    return "", "Valid"


def get_products(search_query=""):
    with connect_db() as conn:
        rows = conn.execute("""
            SELECT barcode, name, price, expiry_date, quantity, min_stock, discount_percent
            FROM products ORDER BY name
        """).fetchall()

    products = []
    total_value = 0.0
    expiry_alerts = 0
    low_stock_count = 0

    for row in rows:
        barcode, name, price, expiry, quantity, min_stock, discount_percent = row
        css, status = status_for(expiry)
        discount_percent = float(discount_percent or 0)
        discount_active = status == "Near expiry" and discount_percent > 0
        discounted_price = round(float(price or 0) * (1 - discount_percent / 100), 2) if discount_active else None
        product = {
            "barcode": barcode,
            "name": name,
            "price": float(price or 0),
            "expiry_date": expiry,
            "quantity": int(quantity or 0),
            "min_stock": int(min_stock if min_stock is not None else 2),
            "discount_percent": discount_percent,
            "discount_active": discount_active,
            "discounted_price": discounted_price,
            "status_css": css,
            "status": status,
        }
        total_value += product["price"] * product["quantity"]
        if status in ("Expired", "Near expiry"):
            expiry_alerts += 1
        if product["quantity"] <= product["min_stock"]:
            low_stock_count += 1
        if search_query and search_query.casefold() not in (
            str(barcode) + " " + str(name)
        ).casefold():
            continue
        products.append(product)

    expiry_notifications = [
        {"name": p["name"], "expiry": p["expiry_date"], "status": p["status"]}
        for p in products if p["status"] in ("Expired", "Near expiry")
    ]
    with connect_db() as conn:
        history = conn.execute("""SELECT barcode, product_name, action, old_quantity,
            new_quantity, note, created_at FROM stock_history ORDER BY id DESC LIMIT 20""").fetchall()
    stock_history = [dict(zip(("barcode", "product_name", "action", "old_quantity", "new_quantity", "note", "created_at"), row)) for row in history]
    return products, total_value, expiry_alerts, low_stock_count, expiry_notifications, stock_history


# ==================== HTML DASHBOARD ====================
PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Shelf | AI Inventory</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--navy:#12243a;--navy2:#1b3552;--teal:#087f8c;--teal2:#0b9aa5;--canvas:#f3f6fa;--surface:#ffffff;--ink:#18283c;--muted:#738196;--line:#e2e8f0;--soft:#e8f6f6;--gold:#b7791f;--red:#c2414b;--green:#18794e;--shadow:0 12px 32px rgba(22,40,64,.07)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--canvas);color:var(--ink);font-family:'DM Sans',Inter,Arial,sans-serif;font-size:14px;line-height:1.5}button,input{font:inherit}button,.btn{cursor:pointer;transition:background .18s ease,transform .18s ease,box-shadow .18s ease}.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:1px solid transparent;border-radius:10px;padding:10px 15px;font-weight:700;text-decoration:none;white-space:nowrap}.btn:hover{transform:translateY(-1px);box-shadow:0 5px 12px #12243a18}.primary{background:var(--teal);color:white}.primary:hover{background:#066a75}.soft{background:#eaf3f7;color:var(--navy);border-color:#dce8ef}.soft:hover{background:#dcecf2}.danger{background:#fff0f0;color:var(--red);border-color:#f9d7d9}.danger:hover{background:#ffe3e4}
.topbar{background:linear-gradient(112deg,#102238 0%,#173b56 60%,#087f8c 140%);color:white;padding:19px max(5%,calc((100% - 1200px)/2));display:flex;align-items:center;justify-content:space-between;gap:18px;box-shadow:0 5px 22px #10223820}.brand{display:flex;align-items:center;gap:13px}.logo{width:48px;height:48px;border-radius:15px;background:linear-gradient(145deg,#e6fbf8,#bfe9e6);color:#087f8c;display:grid;place-items:center;font-size:25px;box-shadow:inset 0 0 0 1px #ffffff88}.brand h1{font:800 24px 'Nunito',sans-serif;margin:0;letter-spacing:-.4px}.brand p{margin:3px 0 0;color:#c9dbe7;font-size:10px;font-weight:700;letter-spacing:1.5px}.live{font-size:12px;color:#d8f7ec;background:#ffffff12;border:1px solid #ffffff25;padding:8px 12px;border-radius:30px}
main{max-width:1240px;margin:30px auto;padding:0 24px 48px}.welcome{display:flex;justify-content:space-between;align-items:end;flex-wrap:wrap;gap:16px;margin:8px 0 22px}h2,h3{font-family:'Nunito',sans-serif;letter-spacing:-.25px}.welcome h2{font-size:29px;line-height:1.2;margin:0;color:var(--navy)}.welcome p{color:var(--muted);margin:8px 0 0;font-size:14px}.panel-head h3{font-size:19px;margin:0;color:var(--navy);font-weight:800}.panel-head{display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap;margin-bottom:18px}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:22px}.stat,.panel{background:var(--surface);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.stat{padding:21px;display:flex;gap:14px;align-items:center;min-width:0;position:relative;overflow:hidden}.stat:after{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--teal)}.stat:nth-child(2):after{background:#6482c3}.stat:nth-child(3):after{background:#d49a36}.stat:nth-child(4):after{background:#d35d65}.stat-icon{width:48px;height:48px;flex:0 0 48px;border-radius:14px;background:#e7f5f5;display:grid;place-items:center;font-size:22px}.stat:nth-child(2) .stat-icon{background:#edf1fb}.stat:nth-child(3) .stat-icon{background:#fff5e4}.stat:nth-child(4) .stat-icon{background:#fff0f0}.stat small{color:var(--muted);display:block;margin-bottom:4px;font-size:12px;font-weight:600}.stat strong{font:800 25px 'Nunito',sans-serif;color:var(--navy);overflow-wrap:anywhere}
.panel{padding:23px;margin-bottom:20px}.search{display:flex;gap:8px;flex-wrap:wrap}.search input{width:min(300px,65vw)}input{min-width:0;border:1px solid #d7e0e9;border-radius:9px;padding:11px 12px;background:#fff;color:var(--ink);width:100%;outline:none;transition:border-color .18s,box-shadow .18s}input:focus{border-color:var(--teal2);box-shadow:0 0 0 3px #0b9aa51c}input::placeholder{color:#9aa7b6}.field label{display:block;font-size:12px;font-weight:800;margin-bottom:7px;color:#465a70}.field small{display:block;margin-top:6px;color:var(--muted);font-size:11px}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}.form-actions{display:flex;gap:10px;margin-top:20px;flex-wrap:wrap}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:separate;border-spacing:0;white-space:nowrap}th{text-align:left;color:#617187;font-size:10px;font-weight:800;letter-spacing:.08em;padding:14px 13px;background:#f6f8fb;border-bottom:1px solid var(--line)}td{padding:14px 13px;border-bottom:1px solid #edf1f5;font-size:12px;vertical-align:middle}tbody tr{background:#fff;transition:background .15s}tbody tr:hover{background:#f8fbfd}tbody tr:last-child td{border-bottom:0}.product{font-weight:800;color:var(--navy);font-size:13px}.code{font-size:11px;color:var(--muted)}.price{font-size:15px;color:var(--teal);font-weight:800}.pill{display:inline-flex;align-items:center;gap:5px;padding:5px 9px;border-radius:30px;font-size:10px;font-weight:800;background:#e5f6ec;color:#167447}.pill.warn{background:#fff3db;color:#99600d}.pill.bad{background:#ffebed;color:#b52d3b}.actions{display:flex;gap:6px;align-items:center}.actions .btn{padding:7px 10px;font-size:11px}.actions form{margin:0}.discount{color:#b7791f;font-weight:900}.history-note{white-space:normal;min-width:170px;max-width:260px;color:#66758a}.notice{padding:13px 15px;border:1px solid #cce9e7;border-radius:11px;background:#effafa;color:#155e63;margin-bottom:11px;font-size:13px}.notice.error{background:#fff0f0;border-color:#f5d1d5;color:#a52f3b}.empty{text-align:center;color:var(--muted);padding:32px!important}footer{text-align:center;color:#8a98a9;font-size:11px;padding:16px 10px;letter-spacing:.2px}
.camera-screen{background:radial-gradient(circle at 50% 30%,#263d52,#101e2d);border:1px solid #263c50;border-radius:13px;overflow:hidden;text-align:center;min-height:220px;display:flex;align-items:center;justify-content:center}.camera-screen img{display:none;width:100%;max-height:470px;object-fit:contain}.camera-placeholder{padding:70px 15px;color:#d6e3ed;font-size:13px}.camera-controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}.camera-controls button:disabled{opacity:.65;cursor:wait}.panel#live-camera-panel{border-top:3px solid var(--teal)}#camera-status{font-size:11px;font-weight:800;color:var(--teal);background:#e8f7f7;padding:6px 10px;border-radius:30px}#ai-results{margin-bottom:0}#ai-results ul{margin:8px 0;padding-left:22px}#ai-results small{display:block;margin-top:8px;color:#68788b}
@media(max-width:1000px){.stats{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:700px){.topbar{padding:15px 18px}.brand h1{font-size:21px}.brand p{font-size:9px;letter-spacing:1px}.live{font-size:10px;padding:7px 9px}main{margin-top:20px;padding:0 13px 30px}.welcome h2{font-size:24px}.panel{padding:15px;border-radius:13px}.stats{grid-template-columns:1fr}.stat{padding:17px}.form-grid{grid-template-columns:1fr}.search{width:100%}.search input{width:100%;flex:1}.panel-head h3{font-size:18px}.camera-controls .btn{flex:1}.table-wrap{border-radius:9px}}
</style>
</head>
<body>
<header class="topbar"><div class="brand"><div class="logo">🌿</div><div><h1>Smart Shelf</h1><p>AI INVENTORY & SHELF MONITORING</p></div></div><div class="live">● Local dashboard</div></header>
<main>
{% with messages = get_flashed_messages(with_categories=true) %}{% for category,message in messages %}<div class="notice {{ 'error' if category=='error' else '' }}">{{ message }}</div>{% endfor %}{% endwith %}
<section class="panel" id="live-camera-panel">
 <div class="panel-head"><h3>Live Shelf Camera</h3><span id="camera-status">Camera stopped</span></div>
 <div class="camera-screen"><img id="live-camera" alt="Live shelf camera preview"><div class="camera-placeholder" id="camera-placeholder">Start the camera to view your shelf</div></div>
 <div class="camera-controls"><button class="btn primary" type="button" onclick="startLiveCamera()">Start Camera</button><button class="btn soft" type="button" onclick="stopLiveCamera()">Stop Camera</button><button class="btn primary" id="ai-scan-btn" type="button" onclick="scanShelfAI()">Scan Shelf with AI</button></div>
 <div id="ai-results" class="notice" style="margin-top:14px">AI detection results will appear here.</div>
</section>
<div class="welcome"><div><h2>{{ 'Edit product' if editing else 'Inventory overview' }}</h2><p>Monitor shelf activity, product availability, expiry risk, and stock value from one dashboard.</p></div><a class="btn primary" href="{{ url_for('dashboard',add=1) if not editing else url_for('dashboard') }}#product-form">＋ Add product</a></div>
<section class="stats">
 <div class="stat"><div class="stat-icon">📦</div><div><small>Registered products</small><strong>{{ products|length }}</strong></div></div>
 <div class="stat"><div class="stat-icon">🪙</div><div><small>Total inventory value</small><strong>₹{{ '%.2f'|format(total_value) }}</strong></div></div>
 <div class="stat"><div class="stat-icon">⏳</div><div><small>Expiry alerts</small><strong>{{ expiry_alerts }}</strong></div></div>
 <div class="stat"><div class="stat-icon">⚠️</div><div><small>Low-stock products</small><strong>{{ low_stock_count }}</strong></div></div>
</section>
<section class="panel"><div class="panel-head"><h3>⚠️ Low Stock Alerts</h3></div>
{% set ns = namespace(found=false) %}{% for p in products %}{% if p.quantity <= p.min_stock %}{% set ns.found=true %}<div class="notice {{ 'error' if p.quantity == 0 else '' }}"><strong>{{ p.name }}</strong> — Quantity: {{ p.quantity }} (Minimum: {{ p.min_stock }}){% if p.quantity == 0 %} · OUT OF STOCK{% endif %}</div>{% endif %}{% endfor %}
{% if not ns.found %}<div class="notice">All products have sufficient stock.</div>{% endif %}
</section>
<section class="panel"><div class="panel-head"><h3>Expiry Alerts</h3></div>
{% set ns2=namespace(found=false) %}{% for p in products %}{% if p.status in ['Expired','Near expiry'] %}{% set ns2.found=true %}<div class="notice {{ 'error' if p.status=='Expired' else '' }}"><strong>{{ p.name }}</strong> — {{ p.status }}{% if p.expiry_date %} (Expiry date: {{ p.expiry_date }}){% endif %}</div>{% endif %}{% endfor %}
{% if not ns2.found %}<div class="notice">No expired or near-expiry products.</div>{% endif %}
</section>
{% if show_form or editing %}
<section class="panel" id="product-form"><div class="panel-head"><h3>{{ 'Update product details' if editing else 'Add a new product' }}</h3></div>
<form method="post" action="{{ url_for('edit_product',barcode=editing.barcode) if editing else url_for('add_product') }}"><div class="form-grid">
<div class="field"><label>Product name *</label><input name="name" required maxlength="100" placeholder="e.g. Pepsodent Paste" value="{{ editing.name if editing else '' }}"></div>
<div class="field"><label>Barcode *</label><input id="barcode" name="barcode" required maxlength="80" placeholder="Scan or enter barcode" value="{{ editing.barcode if editing else '' }}" {{ 'readonly' if editing else '' }}>{% if not editing %}<button class="btn soft" style="margin-top:8px" type="button" id="scan-btn" onclick="scanBarcode()">Scan Barcode</button>{% endif %}</div>
<div class="field"><label>Price (₹) *</label><input name="price" type="number" min="0" step="0.01" required placeholder="20.00" value="{{ editing.price if editing else '' }}"></div>
<div class="field"><label>Expiry date</label><input name="expiry_date" type="date" value="{{ editing.expiry_date if editing and editing.expiry_date else '' }}"></div>
<div class="field"><label>Quantity *</label><input name="quantity" type="number" min="0" step="1" required value="{{ editing.quantity if editing else 1 }}"></div>
<div class="field"><label>Minimum stock quantity *</label><input name="min_stock" type="number" min="0" step="1" required value="{{ editing.min_stock if editing else 2 }}"></div>
<div class="field"><label>Near-expiry discount (%)</label><input name="discount_percent" type="number" min="0" max="90" step="1" value="{{ editing.discount_percent if editing else 0 }}" placeholder="e.g. 10"><small>Discount applies on the dashboard only when the item is near expiry.</small></div>
</div><div class="form-actions"><button class="btn primary" type="submit">{{ 'Save changes' if editing else 'Register product' }}</button><a class="btn soft" href="{{ url_for('dashboard') }}">Cancel</a></div></form></section>
{% endif %}
<section class="panel"><div class="panel-head"><h3>Product inventory</h3><form class="search" method="get" action="{{ url_for('dashboard') }}"><input name="q" value="{{ query }}" placeholder="Search name or barcode"><button class="btn soft" type="submit">Search</button>{% if query %}<a class="btn soft" href="{{ url_for('dashboard') }}">Clear</a>{% endif %}</form></div>
<div class="table-wrap"><table><thead><tr><th>PRODUCT</th><th>BARCODE</th><th>PRICE</th><th>EXPIRY</th><th>QTY</th><th>MIN STOCK</th><th>DISCOUNT</th><th>SELLING PRICE</th><th>STATUS</th><th>ACTIONS</th></tr></thead><tbody>
{% for p in products %}<tr><td><div class="product">{{ p.name }}</div><div class="code">Product item</div></td><td>{{ p.barcode }}</td><td class="price">₹{{ '%.2f'|format(p.price) }}</td><td>{{ p.expiry_date or 'Not entered' }}</td><td>{{ p.quantity }}</td><td>{{ p.min_stock }}</td><td>{% if p.discount_percent > 0 %}{{ p.discount_percent|round(1) }}%{% else %}—{% endif %}</td><td>{% if p.discount_active %}<span class="discount">₹{{ '%.2f'|format(p.discounted_price) }}</span><div class="code">Was ₹{{ '%.2f'|format(p.price) }}</div>{% else %}₹{{ '%.2f'|format(p.price) }}{% if p.status == 'Near expiry' and p.discount_percent == 0 %}<div class="code">Set discount</div>{% endif %}{% endif %}</td><td><span class="pill {{ p.status_css }}">{{ p.status }}</span></td><td><div class="actions"><a class="btn soft" href="{{ url_for('edit_product',barcode=p.barcode) }}">Edit</a><form method="post" action="{{ url_for('delete_product',barcode=p.barcode) }}" onsubmit="return confirm('Delete this product from inventory?')"><button class="btn danger" type="submit">Delete</button></form></div></td></tr>
{% else %}<tr><td colspan="10" class="empty">No products found. Add your first product above.</td></tr>{% endfor %}
</tbody></table></div></section>
<section class="panel"><div class="panel-head"><h3>📜 Stock History</h3><span class="code">Latest 20 stock changes</span></div>
<div class="table-wrap"><table><thead><tr><th>DATE / TIME</th><th>PRODUCT</th><th>BARCODE</th><th>ACTION</th><th>OLD QTY</th><th>NEW QTY</th><th>NOTE</th></tr></thead><tbody>
{% for h in stock_history %}<tr><td>{{ h.created_at }}</td><td>{{ h.product_name }}</td><td>{{ h.barcode }}</td><td>{{ h.action }}</td><td>{{ h.old_quantity if h.old_quantity is not none else '—' }}</td><td>{{ h.new_quantity if h.new_quantity is not none else '—' }}</td><td class="history-note">{{ h.note or '—' }}</td></tr>{% else %}<tr><td colspan="7" class="empty">No stock changes recorded yet.</td></tr>{% endfor %}
</tbody></table></div></section>
<footer>Smart Shelf · Local inventory dashboard · Product data stored on this computer</footer>
</main>
<script>
const CAMERA_URL = {{ camera_url|tojson }};
function startLiveCamera(){
 const img=document.getElementById('live-camera'), placeholder=document.getElementById('camera-placeholder'), status=document.getElementById('camera-status');
 img.onerror=()=>{status.textContent='Camera unavailable';img.style.display='none';placeholder.style.display='block';placeholder.textContent='Camera unavailable. Check IP Webcam is running and phone and laptop share the same network.';};
 img.onload=()=>{status.textContent='Live camera';placeholder.style.display='none';};
 const sep=CAMERA_URL.includes('?')?'&':'?';img.src=CAMERA_URL+sep+'t='+Date.now();img.style.display='block';placeholder.style.display='none';status.textContent='Connecting...';
}
function stopLiveCamera(){const img=document.getElementById('live-camera');img.removeAttribute('src');img.style.display='none';document.getElementById('camera-placeholder').style.display='block';document.getElementById('camera-placeholder').textContent='Camera stopped';document.getElementById('camera-status').textContent='Camera stopped';}
async function scanShelfAI(){
 const button=document.getElementById('ai-scan-btn'), results=document.getElementById('ai-results');button.disabled=true;button.textContent='Scanning shelf...';results.className='notice';results.textContent='Capturing a frame and running the trained AI model...';
 try{const response=await fetch('/detect-shelf',{method:'POST'});const data=await response.json();if(!response.ok)throw new Error(data.error||'AI scan failed.');
 if(!data.detections.length){results.textContent='No trained products detected. Move products closer and improve lighting.';}
 else{results.innerHTML='<strong>Detected products in this frame:</strong><ul>'+data.detections.map(x=>'<li>'+escapeHtml(x.name.replaceAll('_',' '))+' — '+(x.confidence*100).toFixed(1)+'% confidence</li>').join('')+'</ul>';}
 if(data.not_detected&&data.not_detected.length){results.innerHTML+='<div class="notice error"><strong>⚠ Possible missing product(s):</strong> '+data.not_detected.map(x=>escapeHtml(x.replaceAll('_',' '))).join(', ')+'. Please check the shelf and add/restock the product if it is actually missing.</div><small>Confirm with another scan before treating it as missing.</small>';}
 }catch(e){results.className='notice error';results.textContent=e.message;}finally{button.disabled=false;button.textContent='Scan Shelf with AI';}
}
function escapeHtml(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'","&#39;");}
async function scanBarcode(){
 const button=document.getElementById('scan-btn'), status=document.getElementById('ai-results');
 if(!button)return;
 button.disabled=true;button.textContent='Scanning barcode...';status.className='notice';status.textContent='Point the phone camera at the barcode and hold it steady. Scanning up to 10 seconds...';
 try{
   const response=await fetch('/scan-barcode',{method:'POST',cache:'no-store'});
   const data=await response.json();
   if(!response.ok)throw new Error(data.error||'Barcode scan failed.');
   document.getElementById('barcode').value=data.barcode;
   const lookup=await fetch('/product-by-barcode/'+encodeURIComponent(data.barcode),{cache:'no-store'});
   const p=await lookup.json();
   if(lookup.ok){
     document.querySelector('[name=name]').value=p.name;
     document.querySelector('[name=price]').value=p.price;
     document.querySelector('[name=expiry_date]').value=p.expiry_date||'';
     document.querySelector('[name=quantity]').value=p.quantity;
     document.querySelector('[name=min_stock]').value=p.min_stock;
     const discount=document.querySelector('[name=discount_percent]');if(discount)discount.value=p.discount_percent||0;
     status.className='notice';status.textContent='Barcode detected: '+data.barcode+' — existing product details filled in.';
   } else {
     status.className='notice';status.textContent='Barcode detected: '+data.barcode+'. This barcode is not registered yet. Enter the product name, price and other details, then click Register product.';
     document.querySelector('[name=name]').focus();
   }
 }catch(e){status.className='notice error';status.textContent=e.message;}
 finally{button.disabled=false;button.textContent='Scan Barcode';}
}
window.addEventListener('DOMContentLoaded',()=>{startLiveCamera();});
</script>
</body></html>
"""


# ==================== ROUTES ====================
@app.route("/")
def dashboard():
    query = request.args.get("q", "").strip()
    show_form = request.args.get("add") == "1"
    products, total_value, expiry_alerts, low_stock_count, expiry_notifications, stock_history = get_products(query)
    return render_template_string(
        PAGE, products=products, total_value=total_value, expiry_alerts=expiry_alerts,
        low_stock_count=low_stock_count, query=query, show_form=show_form,
        expiry_notifications=expiry_notifications, stock_history=stock_history, editing=None, camera_url=CAMERA_URL
    )


@app.route("/add", methods=["POST"])
def add_product():
    name = request.form.get("name", "").strip()
    barcode = request.form.get("barcode", "").strip()
    expiry = request.form.get("expiry_date", "").strip() or None
    try:
        price = float(request.form.get("price", ""))
        quantity = int(request.form.get("quantity", "1"))
        min_stock = int(request.form.get("min_stock", "2"))
        discount_percent = parse_discount(request.form.get("discount_percent", "0"))
        if not name or not barcode or price < 0 or quantity < 0 or min_stock < 0 or not 0 <= discount_percent <= 90:
            raise ValueError
        if expiry:
            date.fromisoformat(expiry)
        with connect_db() as conn:
            conn.execute("""INSERT INTO products (barcode,name,price,expiry_date,quantity,min_stock,discount_percent)
                            VALUES (?,?,?,?,?,?,?)""", (barcode, name, price, expiry, quantity, min_stock, discount_percent))
            log_stock_change(conn, barcode, name, "PRODUCT ADDED", None, quantity, "Product registered")
        flash("Product registered successfully.")
    except sqlite3.IntegrityError:
        flash("That barcode already exists. Use Edit to update the product.", "error")
    except (ValueError, TypeError):
        flash("Check the name, barcode, price, quantity, minimum stock, expiry date, and discount (0–90%).", "error")
    return redirect(url_for("dashboard"))


@app.route("/edit/<path:barcode>", methods=["GET", "POST"])
def edit_product(barcode):
    with connect_db() as conn:
        row = conn.execute("""SELECT barcode,name,price,expiry_date,quantity,min_stock,discount_percent
                              FROM products WHERE barcode=?""", (barcode,)).fetchone()
    if row is None:
        flash("Product not found.", "error")
        return redirect(url_for("dashboard"))
    editing = {
        "barcode": row[0], "name": row[1], "price": row[2],
        "expiry_date": row[3], "quantity": row[4], "min_stock": row[5], "discount_percent": row[6]
    }
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        expiry = request.form.get("expiry_date", "").strip() or None
        try:
            price = float(request.form.get("price", ""))
            quantity = int(request.form.get("quantity", ""))
            min_stock = int(request.form.get("min_stock", "2"))
            discount_percent = parse_discount(request.form.get("discount_percent", "0"))
            if not name or price < 0 or quantity < 0 or min_stock < 0 or not 0 <= discount_percent <= 90:
                raise ValueError
            if expiry:
                date.fromisoformat(expiry)
            with connect_db() as conn:
                old_quantity = conn.execute("SELECT quantity FROM products WHERE barcode=?", (barcode,)).fetchone()[0]
                conn.execute("""UPDATE products SET name=?,price=?,expiry_date=?,quantity=?,min_stock=?,discount_percent=?
                                WHERE barcode=?""", (name, price, expiry, quantity, min_stock, discount_percent, barcode))
                if old_quantity != quantity:
                    log_stock_change(conn, barcode, name, "QUANTITY UPDATED", old_quantity, quantity, "Quantity changed while editing product")
                else:
                    log_stock_change(conn, barcode, name, "PRODUCT UPDATED", old_quantity, quantity, "Product details or discount updated")
            flash("Product updated successfully.")
            return redirect(url_for("dashboard"))
        except (ValueError, TypeError):
            flash("Check the name, price, quantity, minimum stock, expiry date, and discount (0–90%).", "error")
    products, total_value, expiry_alerts, low_stock_count, expiry_notifications, stock_history = get_products()
    return render_template_string(
        PAGE, products=products, total_value=total_value, expiry_alerts=expiry_alerts,
        low_stock_count=low_stock_count, query="", show_form=False,
        expiry_notifications=expiry_notifications, stock_history=stock_history, editing=editing, camera_url=CAMERA_URL
    )


@app.route("/delete/<path:barcode>", methods=["POST"])
def delete_product(barcode):
    try:
        with connect_db() as conn:
            row = conn.execute("SELECT name, quantity FROM products WHERE barcode=?", (barcode,)).fetchone()
            if not row:
                flash("Product was not found; it may already have been deleted.", "error")
                return redirect(url_for("dashboard"))
            # Delete the product even if a legacy history table has an unusual schema.
            try:
                log_stock_change(conn, barcode, row[0], "PRODUCT DELETED", row[1], None,
                                 "Product removed from inventory")
            except sqlite3.Error:
                app.logger.exception("Could not record delete event in stock_history")
            conn.execute("DELETE FROM products WHERE barcode=?", (barcode,))
        flash("Product deleted successfully.")
    except sqlite3.Error as exc:
        app.logger.exception("Product deletion failed")
        flash("Could not delete product: " + str(exc), "error")
    return redirect(url_for("dashboard"))

@app.route("/scan-barcode", methods=["GET", "POST"])
def scan_barcode():
    """Fast, multi-frame barcode reader using pyzbar plus OpenCV fallbacks."""
    camera = None
    try:
        if decode_barcodes is None and not hasattr(cv2, "barcode_BarcodeDetector") and not hasattr(cv2, "BarcodeDetector"):
            return jsonify({"error": "No barcode decoder is available. Install pyzbar and the Windows ZBar runtime, then restart the app."}), 503

        camera = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_URL)
        if not camera.isOpened():
            return jsonify({"error": "Phone camera is not connected. Start IP Webcam and check CAMERA_URL in app.py."}), 503

        try:
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        detector = None
        try:
            if hasattr(cv2, "barcode_BarcodeDetector"):
                detector = cv2.barcode_BarcodeDetector()
            elif hasattr(cv2, "BarcodeDetector"):
                detector = cv2.BarcodeDetector()
        except Exception:
            detector = None

        def valid_value(value):
            if value is None:
                return ""
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            value = str(value).strip()
            # Reject empty/whitespace results; preserve leading zeros in barcodes.
            return value if value and any(ch.isalnum() for ch in value) else ""

        def try_decode(frame):
            # Try the original image first; transformations are only fallbacks.
            candidates = [frame]
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidates.append(gray)
                # A centered crop reduces background clutter when the barcode is held centrally.
                h, w = gray.shape[:2]
                crop = gray[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
                if crop.size:
                    candidates.append(crop)
                # Upscale small barcodes and improve local contrast for phone-camera frames.
                base = crop if crop.size else gray
                if max(base.shape[:2]) < 1600:
                    scale = min(2.0, 1600 / max(base.shape[:2]))
                    enlarged = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    candidates.append(enlarged)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    candidates.append(clahe.apply(enlarged))
                # Sharpen as another fallback; avoid spending time on many thresholds.
                blur = cv2.GaussianBlur(base, (0, 0), 1.2)
                candidates.append(cv2.addWeighted(base, 1.6, blur, -0.6, 0))
            except Exception:
                pass

            for image in candidates:
                if decode_barcodes is not None:
                    try:
                        decoded = decode_barcodes(image)
                        if decoded:
                            # Prefer EAN/UPC/Code-128 values, preserve leading zeros.
                            item = decoded[0]
                            value = valid_value(item.data)
                            if value:
                                return value, str(getattr(item, "type", "barcode"))
                    except Exception:
                        app.logger.debug("pyzbar decode attempt failed", exc_info=True)

                if detector is not None:
                    try:
                        if hasattr(detector, "detectAndDecodeWithType"):
                            result = detector.detectAndDecodeWithType(image)
                            if isinstance(result, tuple) and len(result) >= 2:
                                values = result[1]
                                if isinstance(values, (list, tuple)):
                                    for value in values:
                                        value = valid_value(value)
                                        if value:
                                            return value, "OpenCV barcode"
                        elif hasattr(detector, "detectAndDecode"):
                            result = detector.detectAndDecode(image)
                            if isinstance(result, tuple) and len(result) >= 2:
                                values = result[1]
                                if isinstance(values, (list, tuple)):
                                    for value in values:
                                        value = valid_value(value)
                                        if value:
                                            return value, "OpenCV barcode"
                                elif valid_value(values):
                                    return valid_value(values), "OpenCV barcode"
                    except Exception:
                        app.logger.debug("OpenCV barcode decode attempt failed", exc_info=True)
            return "", ""

        # Drain a few stale frames, then try up to 35 current frames for a quick scan.
        frame = None
        for _ in range(3):
            ok, candidate = camera.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            return jsonify({"error": "Camera connected but no frame arrived. Check the IP Webcam video stream."}), 503

        for _ in range(35):
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            value, fmt = try_decode(frame)
            if value:
                return jsonify({"barcode": value, "format": fmt, "message": "Barcode scanned successfully."})

        return jsonify({"error": "Barcode not read. Keep the full barcode close, sharp and steady; use bright light, avoid glare, and make sure the barcode bars are horizontal and not cut off."}), 404
    except Exception as exc:
        app.logger.exception("Barcode scan failed")
        return jsonify({"error": "Barcode scan failed: " + str(exc)}), 500
    finally:
        if camera is not None:
            camera.release()

@app.route("/product-by-barcode/<path:barcode>")
def product_by_barcode(barcode):
    with connect_db() as conn:
        row = conn.execute("""SELECT barcode,name,price,expiry_date,quantity,min_stock,discount_percent
                              FROM products WHERE barcode=?""", (barcode,)).fetchone()
    if row is None:
        return jsonify({"error": "Product not found. Please register it first."}), 404
    return jsonify({"barcode": row[0], "name": row[1], "price": row[2],
                    "expiry_date": row[3] or "", "quantity": row[4], "min_stock": row[5], "discount_percent": row[6]})


@app.route("/detect-shelf", methods=["POST"])
def detect_shelf():
    """Run optimized YOLO inference on recent camera frames."""
    if MODEL is None:
        return jsonify({"error": "AI model could not be loaded: " + str(MODEL_LOAD_ERROR)}), 500

    camera = None
    started = datetime.now()
    try:
        # FFMPEG often reads IP Webcam streams more reliably; fall back to default backend.
        camera = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_URL)
        if not camera.isOpened():
            return jsonify({"error": "Phone camera is not connected. Start IP Webcam and check CAMERA_URL."}), 503

        try:
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        frames = []
        # Discard stale buffered frames, then use the newest available frames.
        for _ in range(2):
            ok, frame = camera.read()
            if ok and frame is not None:
                frames.append(frame)
        if not frames:
            return jsonify({"error": "Could not read a frame from the phone camera."}), 503

        # Run YOLO on up to two frames. This improves recall when a frame is blurred,
        # while keeping latency much lower than scanning a long sequence.
        best_by_class = {}
        names = MODEL.names
        for frame in frames:
            result = MODEL.predict(
                source=frame,
                imgsz=DETECT_IMGSZ,
                conf=AI_CONFIDENCE,
                iou=0.45,
                device=INFERENCE_DEVICE,
                max_det=100,
                verbose=False,
                augment=False
            )[0]

            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                # Keep the highest confidence per trained class across the frames.
                if name not in best_by_class or confidence > best_by_class[name]:
                    best_by_class[name] = confidence

        all_names = list(names.values()) if isinstance(names, dict) else list(names)
        elapsed = (datetime.now() - started).total_seconds()
        return jsonify({
            "detections": [
                {"name": name, "confidence": confidence}
                for name, confidence in sorted(best_by_class.items())
            ],
            "not_detected": [name for name in all_names if name not in best_by_class],
            "device": "GPU" if INFERENCE_DEVICE == 0 else "CPU",
             "inference_seconds": round(elapsed, 2),
            "frames_checked": len(frames),
            "effective_fps": round(len(frames) / elapsed, 1) if elapsed > 0 else 0,
            "note": "A product not detected in these frames is only a possible missing product. Verify the shelf before changing stock."
        })
    except Exception as exc:
        app.logger.exception("Shelf AI detection failed")
        return jsonify({"error": "AI detection failed: " + str(exc)}), 500
    finally:
        if camera is not None:
            camera.release()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "database": str(DB), "model_loaded": MODEL is not None,
                    "model_classes": MODEL.names if MODEL is not None else {},
                    "camera_url": CAMERA_URL, "pyzbar_available": decode_barcodes is not None,
                    "barcode_import_error": BARCODE_IMPORT_ERROR,
                    "inference_device": "GPU" if INFERENCE_DEVICE == 0 else "CPU",
                    "confidence": AI_CONFIDENCE, "image_size": DETECT_IMGSZ})


initialize_database()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
