import sqlite3
import yaml
import os



with open("configs/database_config.yaml",    "r", encoding="utf-8") as f:
    database_config    = yaml.safe_load(f)

SQL_DB_PATH = database_config['SQL_DB_PATH']


def get_connection():
    conn = sqlite3.connect(SQL_DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")
    return conn