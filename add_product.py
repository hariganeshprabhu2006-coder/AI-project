import sqlite3

conn = sqlite3.connect("products.db")

conn.execute("""
INSERT OR REPLACE INTO products
(barcode, name, price, expiry_date, quantity)
VALUES (?, ?, ?, ?, ?)
""", ("8901030970405", "Toothpaste", 100.00, None, 1))

conn.commit()
conn.close()

print("Product added successfully!")
print("Barcode: 8901030970405")
print("Name: Toothpaste")
print("Price: ₹100.00 (sample price; update with the actual price)")
