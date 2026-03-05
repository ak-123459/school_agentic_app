import sqlite3
import sys

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "dataschool.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]
    print(f"Deleting table: {table_name}")
    cursor.execute(f"DELETE FROM {table_name};")

conn.commit()
conn.close()

print("✅ All tables deleted successfully!")