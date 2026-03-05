# save as check.py in your project root and run: python check.py
from app.config.config import DB_PATH
import sqlite3, os

print("DB_PATH:", DB_PATH)
print("Exists:", os.path.exists(DB_PATH))

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in cur.fetchall()])
conn.close()