from flask import Flask, request, redirect, url_for, render_template_string, flash, jsonify
import sqlite3
from datetime import date, timedelta
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.secret_key = "smart-shelf-local-key"
DB = "products.db"

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Shelf | Inventory</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
 --green:#174c37; --green2:#287451; --cream:#f6f5ed;
 --mint:#e5efe5; --gold:#d9a441; --ink:#20352b;
 --muted:#758477; --line:#e6e9df; --white:#fff;
 --red:#b94335; --orange:#a96913;
}
*{box-sizing:border-box}
body{margin:0;background:var(--cream);color:var(--ink);
 font-family:'DM Sans',Arial,sans-serif}
button,input{font:inherit}
button,.btn{cursor:pointer}
.topbar{background:var(--green);color:white;padding:20px 5%;
 display:flex;align-items:center;justify-content:space-between;gap:18px}
.brand{display:flex;align-items:center;gap:13px}
.logo{width:45px;height:45px;border-radius:14px;background:#e5efe5;
 color:var(--green);display:grid;place-items:center;font-size:24px}
.brand h1{font:800 23px 'Nunito',sans-serif;margin:0}
.brand p{margin:4px 0 0;color:#d6e4d9;font-size:12px}
.live{font-size:12px;color:#e2efdf}
main{max-width:1200px;margin:30px auto;padding:0 22px 45px}
.welcome{display:flex;justify-content:space-between;align-items:end;
 flex-wrap:wrap;gap:15px;margin-bottom:25px}
h2,h3{font-family:'Nunito',sans-serif}
.welcome h2{font-size:28px;margin:0;color:var(--green)}
.welcome p{color:var(--muted);margin:7px 0 0}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;
 border:0;border-radius:10px;padding:12px 16px;font-weight:700;text-decoration:none}
.primary{background:var(--green2);color:white}
.primary:hover{background:var(--green)}
.soft{background:var(--mint);color:var(--green)}
.danger{background:#fff0ed;color:var(--red)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:22px}
.stat,.panel{background:white;border:1px solid #eceee7;border-radius:17px;
 box-shadow:0 5px 20px #1c3d2710}
.stat{padding:20px;display:flex;gap:14px;align-items:center}
.stat-icon{width:47px;height:47px;border-radius:14px;background:var(--mint);
 display:grid;place-items:center;font-size:23px}
.stat small{color:var(--muted);display:block;margin-bottom:5px}
.stat strong{font:800 25px 'Nunito',sans-serif;color:var(--green)}
.panel{padding:23px;margin-bottom:20px}
.panel-head{display:flex;align-items:center;justify-content:space-between;
 gap:15px;flex-wrap:wrap;margin-bottom:17px}
.panel-head h3{font-size:20px;margin:0;color:var(--green)}
.search{display:flex;gap:8px;flex-wrap:wrap}
input{min-width:0;border:1px solid #dce3d9;border-radius:9px;
 padding:11px 12px;background:white;color:var(--ink);width:100%}
.search input{width:min(300px,65vw)}
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;white-space:nowrap}
th{text-align:left;color:var(--muted);font-size:12px;font-weight:700;
 letter-spacing:.04em;padding:13px 12px;background:#f8f9f5}
td{padding:15px 12px;border-bottom:1px solid var(--line);font-size:13px}
tr:last-child td{border-bottom:0}
.product{font-weight:700;color:var(--green)}
.code{font-size:12px;color:var(--muted)}
.price{font-size:16px;color:var(--green2);font-weight:800}
.pill{display:inline-block;padding:6px 9px;border-radius:20px;
 font-size:11px;font-weight:700;background:var(--mint);color:var(--green2)}
.pill.warn{background:#fff1d6;color:var(--orange)}
.pill.bad{background:#ffebe8;color:var(--red)}
.actions{display:flex;gap:6px}
.actions .btn{padding:8px 10px;font-size:12px}
.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}
.field label{display:block;font-size:12px;font-weight:700;margin-bottom:7px;color:#536657}
.field.full{grid-column:1/-1}
.form-actions{display:flex;gap:10px;margin-top:18px}
.notice{padding:12px 15px;border-radius:10px;background:var(--mint);
 color:var(--green);margin-bottom:16px;font-size:14px}
.notice.error{background:#ffebe8;color:var(--red)}
.empty{text-align:center;color:var(--muted);padding:30px}
footer{text-align:center;color:var(--muted);font-size:12px;padding:10px}
@media(max-width:700px){
 .stats{grid-template-columns:1fr}
 .topbar{padding:16px 20px}
 .welcome h2{font-size:24px}
 main{margin-top:22px;padding:0 13px 30px}
 .panel{padding:15px}
 .form-grid{grid-template-columns:1fr}
 .field.full{grid-column:auto}
}
</style>
</head>
<body>
<header class="topbar">
 <div class="brand">
  <div class="logo">🌿</div>
  <div><h1>Smart Shelf</h1><p>INVENTORY MANAGEMENT SYSTEM</p></div>
 </div>
 <div class="live">● Local dashboard</div>
</header>
<main>
 <section class="panel" id="live-camera-panel" style="display:block;margin-bottom:24px;padding:20px">
  <div class="panel-head">
    <h3>Live Shelf Camera</h3>
    <span id="camera-status">Camera stopped</span>
  </div>
  <div style="background:#17251e;border-radius:12px;overflow:hidden;text-align:center">
    <img id="live-camera" alt="Live shelf camera preview"
         style="display:none;width:100%;max-height:480px;object-fit:contain">
    <div id="camera-placeholder" style="padding:65px 15px;color:white">
      Start the camera to view your shelf
    </div>
  </div>
  <div style="display:flex;gap:10px;margin-top:14px">
    <button class="btn primary" type="button" onclick="startLiveCamera()">Start Camera</button>
    <button class="btn soft" type="button" onclick="stopLiveCamera()">Stop Camera</button>
  </div>
</section>
<div class="welcome">
  <div><h2>{{ 'Edit product' if editing else 'Your inventory, at a glance' }}</h2>
  <p>Keep your products organised, visible and up to date.</p></div>
  <a class="btn primary" href="{{ url_for('dashboard', add=1) if not editing else url_for('dashboard') }}#product-form">＋ Add product</a>
 </div>

 {% with messages = get_flashed_messages(with_categories=true) %}
 {% for category, message in messages %}
 <div class="notice {{ 'error' if category == 'error' else '' }}">{{ message }}</div>
 {% endfor %}
 {% endwith %}

 <section class="stats">
  <div class="stat"><div class="stat-icon">📦</div><div>
   <small>Registered products</small><strong>{{ products|length }}</strong>
  </div></div>
  <div class="stat"><div class="stat-icon">🪙</div><div>
   <small>Total inventory value</small><strong>₹{{ '%.2f'|format(total_value) }}</strong>
  </div></div>
  <div class="stat"><div class="stat-icon">⏳</div><div>
   <small>Expiry alerts</small><strong>{{ expiry_alerts }}</strong>
  </div></div>
 </section>

 <section class="panel">
  <div class="panel-head"><h3>Expiry Alerts</h3></div>
  {% set alert_products = products | selectattr(6, 'in', ['Expired', 'Near expiry']) | list %}
  {% if alert_products %}
    {% for p in alert_products %}
      <div class="notice {{ 'error' if p[6] == 'Expired' else '' }}">
        <strong>{{ p[1] }}</strong> ? {{ p[6] }}
        {% if p[3] %}(Expiry date: {{ p[3] }}){% endif %}
      </div>
    {% endfor %}
  {% else %}
    <div class="notice">No expired or near-expiry products.</div>
  {% endif %}
</section>

{% if show_form or editing %}
 <section class="panel" id="product-form">
  <div class="panel-head"><h3>{{ 'Update product details' if editing else 'Add a new product' }}</h3></div>
  <form method="post" action="{{ url_for('edit_product', barcode=editing[0]) if editing else url_for('add_product') }}">
   <div class="form-grid">
    <div class="field"><label>Product name *</label>
     <input name="name" required maxlength="100" placeholder="e.g. Pepsodent Paste" value="{{ editing[1] if editing else '' }}">
    </div>
    <div class="field"><label>Barcode *</label>
     <div style="display:flex;gap:8px;align-items:center">
     <input id="barcode" name="barcode" required maxlength="80" placeholder="Scan or enter barcode" value="{{ editing[0] if editing else '' }}" {{ 'readonly' if editing else '' }}>
     {% if not editing %}<button class="btn soft" type="button" id="scan-btn" onclick="scanBarcode()">Scan Barcode</button>{% endif %}
    </div>
    </div>
    <div class="field"><label>Price (₹) *</label>
     <input name="price" type="number" min="0" step="0.01" required placeholder="20.00" value="{{ editing[2] if editing else '' }}">
    </div>
    <div class="field"><label>Expiry date</label>
     <input name="expiry_date" type="date" value="{{ editing[3] if editing and editing[3] else '' }}">
    </div>
    <div class="field"><label>Quantity *</label>
     <input name="quantity" type="number" min="0" step="1" required value="{{ editing[4] if editing else 1 }}">
    </div>
   </div>
   <div class="form-actions">
    <button class="btn primary" type="submit">{{ 'Save changes' if editing else 'Register product' }}</button>
    <a class="btn soft" href="{{ url_for('dashboard') }}">Cancel</a>
   </div>
  </form>
 </section>
 {% endif %}

 <section class="panel">
  <div class="panel-head">
   <h3>Product inventory</h3>
   <form class="search" method="get" action="{{ url_for('dashboard') }}">
    <input name="q" value="{{ query }}" placeholder="Search name or barcode">
    <button class="btn soft" type="submit">Search</button>
    {% if query %}<a class="btn soft" href="{{ url_for('dashboard') }}">Clear</a>{% endif %}
   </form>
  </div>
  <div class="table-wrap"><table>
   <thead><tr><th>PRODUCT</th><th>BARCODE</th><th>PRICE</th><th>EXPIRY</th><th>QTY</th><th>STATUS</th><th>ACTIONS</th></tr></thead>
   <tbody>
   {% for p in products %}
   <tr>
    <td><div class="product">{{ p[1] }}</div><div class="code">Product item</div></td>
    <td>{{ p[0] }}</td><td class="price">₹{{ '%.2f'|format(p[2]) }}</td>
    <td>{{ p[3] or 'Not entered' }}</td><td>{{ p[4] }}</td>
    <td><span class="pill {{ p[5] }}">{{ p[6] }}</span></td>
    <td><div class="actions">
     <a class="btn soft" href="{{ url_for('edit_product', barcode=p[0]) }}">Edit</a>
     <form method="post" action="{{ url_for('delete_product', barcode=p[0]) }}" onsubmit="return confirm('Delete {{ p[1] }} from inventory?')">
      <button class="btn danger" type="submit">Delete</button>
     </form>
    </div></td>
   </tr>
   {% else %}
   <tr><td colspan="7" class="empty">No products found. Add your first product above.</td></tr>
   {% endfor %}
   </tbody>
  </table></div>
 </section>
 <footer>Smart Shelf · Local inventory dashboard · Product data stored on this computer</footer>
</main>
<script>
async function scanBarcode() {
    const button = document.getElementById("scan-btn");
    const input = document.getElementById("barcode");
    const panel = document.getElementById("live-camera-panel");
    const img = document.getElementById("live-camera");
    const placeholder = document.getElementById("camera-placeholder");
    const status = document.getElementById("camera-status");

    const nameField = document.querySelector('input[name="name"]');
    const priceField = document.querySelector('input[name="price"]');
    const expiryField = document.querySelector('input[name="expiry_date"]');
    const quantityField = document.querySelector('input[name="quantity"]');

    button.disabled = true;
    button.textContent = "Scanning...";
    panel.style.display = "block";
    img.style.display = "block";
    placeholder.style.display = "none";
    status.textContent = "Opening camera and scanning...";
    img.src = "http://10.211.107.39:8080/video?t=" + Date.now();
    panel.scrollIntoView({ behavior: "smooth", block: "center" });

    try {
        await new Promise(resolve => setTimeout(resolve, 1200));

        const response = await fetch("/scan-barcode");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Barcode scan failed.");

        input.value = data.barcode;

        const lookup = await fetch("/product-by-barcode/" + encodeURIComponent(data.barcode));
        const product = await lookup.json();

        if (lookup.ok) {
            nameField.value = product.name;
            priceField.value = product.price;
            expiryField.value = product.expiry_date || "";
            quantityField.value = product.quantity;
            alert("Product found: " + product.name);
        } else {
            alert("Barcode scanned: " + data.barcode +
                  ". This product is not registered yet. Enter its details.");
        }

        img.removeAttribute("src");
        img.style.display = "none";
        panel.style.display = "none";
    } catch (error) {
        status.textContent = "Scan failed — camera preview remains open";
        alert(error.message);
    } finally {
        button.disabled = false;
        button.textContent = "Scan Barcode";
    }
}
</script>
<script>
function startLiveCamera() {
    const img = document.getElementById("live-camera");
    img.src = "http://10.211.107.39:8080/video";
    img.style.display = "block";
    document.getElementById("camera-placeholder").style.display = "none";
    document.getElementById("camera-status").textContent = "Connecting...";
    img.onload = () => {
        document.getElementById("camera-status").textContent = "Live camera";
    };
    img.onerror = () => {
        document.getElementById("camera-status").textContent = "Camera unavailable";
        document.getElementById("camera-placeholder").style.display = "block";
        document.getElementById("camera-placeholder").textContent =
            "Camera unavailable. Check that IP Webcam is running and your phone is connected to the same Wi-Fi.";
    };
}
function stopLiveCamera() {
    const img = document.getElementById("live-camera");
    img.removeAttribute("src");
    img.style.display = "none";
    document.getElementById("camera-placeholder").style.display = "block";
    document.getElementById("camera-placeholder").textContent = "Camera stopped";
    document.getElementById("camera-status").textContent = "Camera stopped";
}
window.addEventListener("DOMContentLoaded", function () {
    startLiveCamera();
});
</script>

<script>
window.addEventListener("DOMContentLoaded", function () {
    const expiryNotifications = {{ expiry_notifications | tojson }};
    if (!expiryNotifications.length) return;

    const lines = expiryNotifications.map(function (item) {
        return item.name + " — " + item.status +
            (item.expiry ? " (Expiry date: " + item.expiry + ")" : "");
    });

    window.setTimeout(function () {
        alert("SMART SHELF EXPIRY NOTIFICATION\n\n" +
              "Please check these products:\n\n" + lines.join("\n"));
    }, 700);
});
</script>

</body>
</html>
"""

def connect_db():
    return sqlite3.connect(DB)

def status_for(expiry):
    if not expiry:
        return ("", "No expiry date")
    try:
        expiry_day = date.fromisoformat(expiry)
    except ValueError:
        return ("", "Check expiry date")
    if expiry_day < date.today():
        return ("bad", "Expired")
    if expiry_day <= date.today() + timedelta(days=30):
        return ("warn", "Near expiry")
    return ("", "Valid")

@app.route("/")
def dashboard():
    query = request.args.get("q", "").strip()
    show_form = request.args.get("add") == "1"
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT barcode, name, price, expiry_date, quantity FROM products ORDER BY name"
        ).fetchall()
    products = []
    total_value = 0
    expiry_alerts = 0
    for p in rows:
        css, status = status_for(p[3])
        if status in ("Expired", "Near expiry"):
            expiry_alerts += 1
        total_value += p[2] * p[4]
        if query and query.casefold() not in (p[0] + " " + p[1]).casefold():
            continue
        products.append((*p, css, status))
    expiry_notifications = [
        {"name": p[1], "expiry": p[3], "status": status_for(p[3])[1]}
        for p in rows
        if status_for(p[3])[1] in ("Expired", "Near expiry")
    ]
    return render_template_string(
        PAGE, products=products, total_value=total_value,
        expiry_alerts=expiry_alerts, query=query, show_form=show_form,
        expiry_notifications=expiry_notifications, editing=None
    )

@app.route("/add", methods=["POST"])
def add_product():
    name = request.form.get("name", "").strip()
    barcode = request.form.get("barcode", "").strip()
    expiry = request.form.get("expiry_date", "").strip() or None
    try:
        price = float(request.form.get("price", ""))
        quantity = int(request.form.get("quantity", "1"))
        if not name or not barcode or price < 0 or quantity < 0:
            raise ValueError()
        if expiry:
            date.fromisoformat(expiry)
        with connect_db() as conn:
            conn.execute(
                "INSERT INTO products (barcode,name,price,expiry_date,quantity) VALUES (?,?,?,?,?)",
                (barcode, name, price, expiry, quantity)
            )
        flash("Product registered successfully.")
    except sqlite3.IntegrityError:
        flash("That barcode already exists. Use Edit to update the product.", "error")
    except (ValueError, TypeError):
        flash("Check the price, quantity and date fields.", "error")
    return redirect(url_for("dashboard"))

@app.route("/edit/<path:barcode>", methods=["GET", "POST"])
def edit_product(barcode):
    with connect_db() as conn:
        old = conn.execute(
            "SELECT barcode,name,price,expiry_date,quantity FROM products WHERE barcode=?",
            (barcode,)
        ).fetchone()
    if not old:
        flash("Product not found.", "error")
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        expiry = request.form.get("expiry_date", "").strip() or None
        try:
            price = float(request.form.get("price", ""))
            quantity = int(request.form.get("quantity", ""))
            if not name or price < 0 or quantity < 0:
                raise ValueError()
            if expiry:
                date.fromisoformat(expiry)
            with connect_db() as conn:
                conn.execute(
                    "UPDATE products SET name=?,price=?,expiry_date=?,quantity=? WHERE barcode=?",
                    (name, price, expiry, quantity, barcode)
                )
            flash("Product updated successfully.")
            return redirect(url_for("dashboard"))
        except (ValueError, TypeError):
            flash("Check the price, quantity and date fields.", "error")
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT barcode,name,price,expiry_date,quantity FROM products ORDER BY name"
        ).fetchall()
    products = [(*p, *status_for(p[3])) for p in rows]
    total_value = sum(p[2] * p[4] for p in rows)
    expiry_alerts = sum(1 for p in rows if status_for(p[3])[1] in ("Expired","Near expiry"))
    expiry_notifications = [
        {"name": p[1], "expiry": p[3], "status": status_for(p[3])[1]}
        for p in rows
        if status_for(p[3])[1] in ("Expired", "Near expiry")
    ]
    return render_template_string(
        PAGE, products=products, total_value=total_value,
        expiry_alerts=expiry_alerts, query="", show_form=False,
        expiry_notifications=expiry_notifications, editing=old
    )

@app.route("/delete/<path:barcode>", methods=["POST"])
def delete_product(barcode):
    with connect_db() as conn:
        conn.execute("DELETE FROM products WHERE barcode=?", (barcode,))
    flash("Product deleted.")
    return redirect(url_for("dashboard"))

from flask import jsonify
from pyzbar.pyzbar import decode
import cv2

@app.route("/scan-barcode")
def scan_barcode():
    camera = cv2.VideoCapture("http://10.211.107.39:8080/video")

    if not camera.isOpened():
        return jsonify({"error": "Phone camera is not connected"}), 503

    try:
        for _ in range(5):
            camera.read()

        for _ in range(30):
            success, frame = camera.read()
            if not success:
                continue

            barcodes = decode(frame)
            if barcodes:
                code = barcodes[0].data.decode("utf-8")
                return jsonify({"barcode": code})

        return jsonify({"error": "No barcode detected. Try again."}), 404
    finally:
        camera.release()


@app.route("/product-by-barcode/<path:barcode>")
def product_by_barcode(barcode):
    row = None
    with connect_db() as conn:
        row = conn.execute(
            "SELECT barcode, name, price, expiry_date, quantity FROM products WHERE barcode = ?",
            (barcode,)
        ).fetchone()

    if row is None:
        return jsonify({"error": "Product not found. Please register it first."}), 404

    return jsonify({
        "barcode": row[0],
        "name": row[1],
        "price": row[2],
        "expiry_date": row[3] or "",
        "quantity": row[4]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)







