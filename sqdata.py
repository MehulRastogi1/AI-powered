# =======================
#   sqdata.py
# =======================

import sqlite3
from datetime import datetime

DB_NAME = "db.sqlite"


# ----------------------------
# CONNECT FUNCTION
# ----------------------------
def get_connection():
    return sqlite3.connect(DB_NAME)


# ----------------------------
# CREATE TABLES
# ----------------------------
def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                username TEXT,
                password TEXT,
                created_at TEXT
        )
    """)

    # Logs table (login / logout tracking)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            login_time TEXT,
            logout_time TEXT
        )
    """)

    # Contact table  ✅ NEW
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contact (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            username TEXT,
            email TEXT,
            message TEXT,
            submitted_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# ----------------------------
# ADD USER
# ----------------------------
def add_user(userid,username, password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            "INSERT INTO users(user_id,username, password, created_at) VALUES (?, ?, ?,?)",
            (userid,username, password, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True

    except sqlite3.IntegrityError:
        conn.close()
        return False   # Username already exists


# ----------------------------
# VERIFY USER LOGIN
# ----------------------------
def verify_user(userid, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE user_id=? AND password=?",
        (userid, password)
    )
    user = cur.fetchone()
    conn.close()

    return user is not None


# ----------------------------
# GET_USERNAME
# ----------------------------
def get_username_from_user_id(user_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute(
        "SELECT username FROM users WHERE user_id = ?",
        (user_id,)
    )
    row = cur.fetchone()
    conn.close()
    
    if row and row[0]:
        return row[0]
    return "Unknown User"

# ----------------------------
# LOG LOGIN TIME
# ----------------------------
def log_login(username):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO logs(username, login_time, logout_time) VALUES (?, ?, ?)",
        (username, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), None)
    )

    conn.commit()

    log_id = cur.lastrowid
    conn.close()

    return log_id  # return id so we can update logout later


# ----------------------------
# LOG LOGOUT TIME
# ----------------------------
def log_logout(log_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "UPDATE logs SET logout_time=? WHERE log_id=?",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), log_id)
    )

    conn.commit()
    conn.close()


# ----------------------------
# ADD CONTACT MESSAGE  ✅ NEW
# ----------------------------
def add_contact(name, username, email, message):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO contact(name, username, email, message, submitted_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        username,
        email,
        message,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()
    return True

