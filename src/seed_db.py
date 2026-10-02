import os
import sys
import json
import sqlite3
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from db_service import get_connection, init_db

def seed_database():
    """Seeds the SQLite database with mock data to populate the dashboards."""
    
    # Ensure DB is initialized
    init_db()
    
    conn = get_connection()
    c = conn.cursor()
    
    now = datetime.now()
    
    print("Seeding Users...")
    users = [
        ("ADM001", "j.admin", "James Admin", "admin", "enrolled", "active", (now - timedelta(days=30)).isoformat(), now.isoformat(), "[]"),
        ("SEC002", "s.guard", "Sarah Security", "security", "enrolled", "active", (now - timedelta(days=15)).isoformat(), (now - timedelta(hours=2)).isoformat(), "[]"),
        ("EMP003", "m.smith", "Michael Smith", "employee", "pending", "active", (now - timedelta(days=5)).isoformat(), None, None)
    ]
    
    c.executemany("""
        INSERT OR IGNORE INTO users (user_id, username, full_name, role, auth_status, account_status, created_at, last_login, face_embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, users)

    print("Seeding Sessions...")
    session_id_admin = "sess_" + str(random.randint(10000, 99999))
    session_id_sec = "sess_" + str(random.randint(10000, 99999))
    
    sessions = [
        (session_id_admin, "ADM001", "192.168.1.50", (now - timedelta(minutes=45)).isoformat(), None),
        (session_id_sec, "SEC002", "192.168.1.101", (now - timedelta(hours=2)).isoformat(), None)
    ]
    c.executemany("""
        INSERT OR IGNORE INTO sessions (session_id, user_id, source_ip, login_time, logout_time)
        VALUES (?, ?, ?, ?, ?)
    """, sessions)

    print("Seeding Auth Logs...")
    auth_logs = [
        ("FACE_AUTH_SUCCESS", "ADM001", "James Admin", 0.96, "SUCCESS", "192.168.1.50", session_id_admin, (now - timedelta(minutes=45)).isoformat()),
        ("UNKNOWN_FACE", None, None, 0.42, "ALERT", "Camera-FrontDoor", None, (now - timedelta(hours=1)).isoformat()),
        ("FACE_AUTH_SUCCESS", "SEC002", "Sarah Security", 0.91, "SUCCESS", "192.168.1.101", session_id_sec, (now - timedelta(hours=2)).isoformat()),
        ("ENROLLMENT", "SEC002", "Sarah Security", None, "SUCCESS", "Local System", None, (now - timedelta(days=15)).isoformat()),
        ("FACE_AUTH_FAIL", "EMP003", "Michael Smith", 0.35, "REJECTED", "Camera-SideDoor", None, (now - timedelta(days=1)).isoformat())
    ]
    c.executemany("""
        INSERT INTO auth_logs (event_type, user_id, full_name, confidence, status, source_ip, session_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, auth_logs)

    print("Seeding Security Events (Attacks)...")
    
    # Dummy SHAP Data
    shap_data_str = json.dumps({
        "base_value": 0.25,
        "features": [
            {"name": "Total Length of Fwd Packets", "value": 15000, "contribution": 1.25},
            {"name": "Fwd Packet Length Max", "value": 1200, "contribution": 0.85},
            {"name": "Flow Bytes/s", "value": 85000, "contribution": 0.45},
            {"name": "Bwd Packet Length Min", "value": 0, "contribution": -0.15}
        ]
    })
    
    sec_events = [
        ("NETWORK_ATTACK", "192.168.1.50", "10.0.0.5", "DDoS / Flooding", 0.98, "CRITICAL", "RESOLVED", "Massive anomalous traffic detected from authenticated IP.", shap_data_str, "ADM001", session_id_admin, (now - timedelta(minutes=5)).isoformat()),
        ("PORT_SCAN", "192.168.1.101", "multiple", "Port Scan", 1.0, "HIGH", "ACTIVE", "Scanning 25 unique ports in under 10 seconds.", None, "SEC002", session_id_sec, (now - timedelta(minutes=15)).isoformat()),
        ("UNKNOWN_FACE", "Camera-FrontDoor", None, "Physical Intrusion", 0.88, "MEDIUM", "ACTIVE", "Unrecognized individual loitering near front entrance.", None, None, None, (now - timedelta(hours=1)).isoformat()),
        ("NETWORK_ATTACK", "145.22.1.9", "192.168.1.1", "Web Attack", 0.92, "HIGH", "RESOLVED", "External SQL injection attempt.", None, None, None, (now - timedelta(days=1)).isoformat()),
    ]
    c.executemany("""
        INSERT INTO security_events (event_type, source_ip, destination_ip, attack_type, confidence, severity, status, description, shap_data, user_id, session_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sec_events)

    conn.commit()
    conn.close()
    print("Database seeded successfully! Refresh your dashboard.")

if __name__ == "__main__":
    seed_database()
