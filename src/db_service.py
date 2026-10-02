"""
db_service.py
--------------
SQLite database service for the AI Security Platform.
Handles: users, sessions, auth logs, security events, system settings.

DESIGN NOTES:
- Uses SQLite for zero-config local deployment
- Face embeddings stored as JSON blobs (never exposed via API responses)
- All timestamps in ISO format UTC
- Thread-safe via check_same_thread=False + connection-per-call pattern
"""

import sqlite3
import json
import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = os.environ.get("DATABASE_PATH", "data/security.db")


def get_connection() -> sqlite3.Connection:
    """Return a new connection with row_factory for dict-like access."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    try:
        c = conn.cursor()

        # ---------- Admin / System Users ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                last_login  TEXT
            )
        """)

        # ---------- Enrolled Face Users ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT UNIQUE NOT NULL,
                username        TEXT UNIQUE NOT NULL,
                full_name       TEXT NOT NULL,
                role            TEXT DEFAULT 'employee',
                face_embedding  TEXT,           -- JSON blob, never returned to frontend
                created_at      TEXT DEFAULT (datetime('now')),
                last_login      TEXT,
                auth_status     TEXT DEFAULT 'pending',
                account_status  TEXT DEFAULT 'active'
            )
        """)

        # ---------- Security Sessions ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT UNIQUE NOT NULL,
                user_id     TEXT NOT NULL,
                login_time  TEXT NOT NULL,
                logout_time TEXT,
                source_ip   TEXT,
                is_active   INTEGER DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        # ---------- Authentication Logs ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS auth_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                user_id     TEXT,
                full_name   TEXT,
                event_type  TEXT NOT NULL,   -- FACE_AUTH_SUCCESS, FACE_AUTH_FAIL, UNKNOWN_FACE, LOGOUT
                confidence  REAL,
                source_ip   TEXT,
                session_id  TEXT,
                status      TEXT             -- SUCCESS, FAILED, ALERT
            )
        """)

        # ---------- Security Events (unified timeline) ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                event_type      TEXT NOT NULL,
                user_id         TEXT,
                session_id      TEXT,
                source_ip       TEXT,
                destination_ip  TEXT,
                attack_type     TEXT,
                confidence      REAL,
                severity        TEXT DEFAULT 'LOW',  -- LOW, MEDIUM, HIGH, CRITICAL
                status          TEXT DEFAULT 'OPEN',
                description     TEXT,
                shap_data       TEXT    -- JSON: top SHAP features for this event
            )
        """)

        # ---------- Blocked IPs (DB mirror of CSV for richer queries) ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS blocked_ips_db (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ip          TEXT UNIQUE NOT NULL,
                reason      TEXT,
                added_by    TEXT DEFAULT 'system',
                added_at    TEXT DEFAULT (datetime('now')),
                status      TEXT DEFAULT 'BLOCKED'
            )
        """)

        # ---------- System Settings ----------
        c.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key     TEXT PRIMARY KEY,
                value   TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # ---------- Default settings ----------
        defaults = {
            "auto_block_enabled": "false",
            "face_match_threshold": "0.50",
            "yolo_confidence": "0.50",
            "session_timeout_minutes": "60",
            "scan_port_threshold": "20",
        }
        for k, v in defaults.items():
            c.execute(
                "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?,?)", (k, v)
            )

        conn.commit()
        _ensure_default_admin(conn)
    finally:
        conn.close()


# ─────────────────────────── Admin Auth ───────────────────────────

def _hash_password(password: str) -> str:
    """SHA-256 + salt. Use werkzeug's generate_password_hash in production."""
    from werkzeug.security import generate_password_hash
    return generate_password_hash(password)


def _verify_password(password: str, hashed: str) -> bool:
    from werkzeug.security import check_password_hash
    return check_password_hash(hashed, password)


def _ensure_default_admin(conn: sqlite3.Connection):
    """Create the default admin account if none exists."""
    username = os.environ.get("ADMIN_USERNAME", "admin")
    password = os.environ.get("ADMIN_PASSWORD", "admin123")
    c = conn.cursor()
    existing = c.execute(
        "SELECT id FROM admin_users WHERE username=?", (username,)
    ).fetchone()
    if not existing:
        c.execute(
            "INSERT INTO admin_users (username, password_hash) VALUES (?,?)",
            (username, _hash_password(password))
        )
        conn.commit()


def validate_admin(username: str, password: str) -> bool:
    """Return True if credentials match a valid admin account."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT password_hash FROM admin_users WHERE username=?", (username,)
        ).fetchone()
        if row and _verify_password(password, row["password_hash"]):
            conn.execute(
                "UPDATE admin_users SET last_login=? WHERE username=?",
                (datetime.utcnow().isoformat(), username)
            )
            conn.commit()
            return True
        return False
    finally:
        conn.close()


# ─────────────────────────── Users ───────────────────────────

def create_user(user_id: str, username: str, full_name: str, role: str = "employee") -> bool:
    """Create a user record. Returns False if user_id or username already exists."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (user_id, username, full_name, role) VALUES (?,?,?,?)",
            (user_id, username, full_name, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def save_face_embedding(user_id: str, embedding: List[float]) -> bool:
    """Store face embedding (as JSON). Returns False if user not found."""
    conn = get_connection()
    try:
        c = conn.cursor()
        result = c.execute("SELECT id FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not result:
            return False
        c.execute(
            "UPDATE users SET face_embedding=?, auth_status='enrolled' WHERE user_id=?",
            (json.dumps(embedding), user_id)
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_all_users(include_embedding: bool = False) -> List[Dict]:
    """Return all users. Embeddings excluded by default for safety."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT user_id, username, full_name, role, created_at, last_login, "
            "auth_status, account_status FROM users"
        ).fetchall()
        result = [dict(r) for r in rows]
        return result
    finally:
        conn.close()


def get_user(user_id: str) -> Optional[Dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, username, full_name, role, created_at, last_login, "
            "auth_status, account_status FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_face_embeddings() -> Dict[str, Dict]:
    """Return {user_id: {embedding, full_name, role}} for recognition loop."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT user_id, full_name, role, face_embedding FROM users "
            "WHERE face_embedding IS NOT NULL AND account_status='active'"
        ).fetchall()
        result = {}
        for r in rows:
            result[r["user_id"]] = {
                "full_name": r["full_name"],
                "role": r["role"],
                "embedding": json.loads(r["face_embedding"]),
            }
        return result
    finally:
        conn.close()


def deactivate_user(user_id: str) -> bool:
    conn = get_connection()
    try:
        c = conn.execute(
            "UPDATE users SET account_status='inactive' WHERE user_id=?", (user_id,)
        )
        conn.commit()
        return c.rowcount > 0
    finally:
        conn.close()


# ─────────────────────────── Sessions ───────────────────────────

def create_session(user_id: str, source_ip: str = None) -> str:
    """Create and return a new session_id."""
    session_id = f"S-{secrets.token_hex(8).upper()}"
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, login_time, source_ip) VALUES (?,?,?,?)",
            (session_id, user_id, datetime.utcnow().isoformat(), source_ip)
        )
        conn.execute(
            "UPDATE users SET last_login=? WHERE user_id=?",
            (datetime.utcnow().isoformat(), user_id)
        )
        conn.commit()
        return session_id
    finally:
        conn.close()


def end_session(session_id: str):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE sessions SET is_active=0, logout_time=? WHERE session_id=?",
            (datetime.utcnow().isoformat(), session_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_active_sessions() -> List[Dict]:
    conn = get_connection()
    try:
        timeout = int(os.environ.get("SESSION_TIMEOUT_MINUTES", 60))
        cutoff = (datetime.utcnow() - timedelta(minutes=timeout)).isoformat()
        rows = conn.execute(
            "SELECT s.*, u.full_name, u.role FROM sessions s "
            "LEFT JOIN users u ON s.user_id=u.user_id "
            "WHERE s.is_active=1 AND s.login_time > ?", (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_session_by_ip(source_ip: str) -> Optional[Dict]:
    """Find the active session associated with a source IP."""
    conn = get_connection()
    try:
        timeout = int(os.environ.get("SESSION_TIMEOUT_MINUTES", 60))
        cutoff = (datetime.utcnow() - timedelta(minutes=timeout)).isoformat()
        row = conn.execute(
            "SELECT s.*, u.full_name, u.role FROM sessions s "
            "LEFT JOIN users u ON s.user_id=u.user_id "
            "WHERE s.source_ip=? AND s.is_active=1 AND s.login_time > ? "
            "ORDER BY s.login_time DESC LIMIT 1",
            (source_ip, cutoff)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ─────────────────────────── Auth Logs ───────────────────────────

def log_auth_event(event_type: str, user_id: str = None, full_name: str = None,
                   confidence: float = None, source_ip: str = None,
                   session_id: str = None, status: str = "SUCCESS"):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO auth_logs (timestamp, user_id, full_name, event_type, "
            "confidence, source_ip, session_id, status) VALUES (?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), user_id, full_name, event_type,
             confidence, source_ip, session_id, status)
        )
        conn.commit()
    finally:
        conn.close()


def get_auth_logs(limit: int = 100) -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM auth_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────── Security Events ───────────────────────────

def log_security_event(event_type: str, source_ip: str = None, destination_ip: str = None,
                       attack_type: str = None, confidence: float = None,
                       severity: str = "LOW", description: str = None,
                       user_id: str = None, session_id: str = None,
                       shap_data: Dict = None) -> int:
    """Insert a security event and return its ID."""
    conn = get_connection()
    try:
        c = conn.execute(
            "INSERT INTO security_events (timestamp, event_type, user_id, session_id, "
            "source_ip, destination_ip, attack_type, confidence, severity, description, shap_data) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), event_type, user_id, session_id,
             source_ip, destination_ip, attack_type, confidence, severity, description,
             json.dumps(shap_data) if shap_data else None)
        )
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()


def get_security_events(limit: int = 100) -> List[Dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM security_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("shap_data"):
                try:
                    d["shap_data"] = json.loads(d["shap_data"])
                except Exception:
                    d["shap_data"] = None
            result.append(d)
        return result
    finally:
        conn.close()


def get_security_event(event_id: int) -> Optional[Dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM security_events WHERE id=?", (event_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("shap_data"):
            try:
                d["shap_data"] = json.loads(d["shap_data"])
            except Exception:
                d["shap_data"] = None
        return d
    finally:
        conn.close()


def update_event_shap(event_id: int, shap_data: Dict):
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE security_events SET shap_data=?, status='ANALYZED' WHERE id=?",
            (json.dumps(shap_data), event_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_dashboard_stats() -> Dict:
    conn = get_connection()
    try:
        users = conn.execute("SELECT COUNT(*) as c FROM users WHERE account_status='active'").fetchone()["c"]
        sessions = conn.execute("SELECT COUNT(*) as c FROM sessions WHERE is_active=1").fetchone()["c"]
        threats = conn.execute("SELECT COUNT(*) as c FROM security_events WHERE event_type NOT IN ('FACE_AUTH_SUCCESS','FACE_AUTH_FAIL')").fetchone()["c"]
        unknown_faces = conn.execute("SELECT COUNT(*) as c FROM security_events WHERE event_type='UNKNOWN_FACE'").fetchone()["c"]
        blocked = conn.execute("SELECT COUNT(*) as c FROM blocked_ips_db WHERE status='BLOCKED'").fetchone()["c"]
        auth_success = conn.execute("SELECT COUNT(*) as c FROM auth_logs WHERE event_type='FACE_AUTH_SUCCESS'").fetchone()["c"]
        auth_fail = conn.execute("SELECT COUNT(*) as c FROM auth_logs WHERE event_type='FACE_AUTH_FAIL' OR event_type='UNKNOWN_FACE'").fetchone()["c"]
        return {
            "total_users": users,
            "active_sessions": sessions,
            "threats_detected": threats,
            "unknown_faces": unknown_faces,
            "blocked_ips": blocked,
            "auth_success": auth_success,
            "auth_fail": auth_fail,
        }
    finally:
        conn.close()


# ─────────────────────────── Settings ───────────────────────────

def get_setting(key: str, default: str = None) -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM system_settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def set_setting(key: str, value: str):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO system_settings (key, value, updated_at) VALUES (?,?,?)",
            (key, value, datetime.utcnow().isoformat())
        )
        conn.commit()
    finally:
        conn.close()


def get_all_settings() -> Dict:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, value FROM system_settings").fetchall()
        return {r["key"]: r["value"] for r in rows}
    finally:
        conn.close()
