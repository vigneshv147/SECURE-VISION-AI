"""
app.py (EXTENDED)
-----------------
Unified AI Security Platform — Flask application.

EXISTING ROUTES (all preserved, unchanged):
  GET  /                       → Dashboard (now SOC dashboard for authenticated admins)
  GET,POST /blocked-ips        → Blocklist management
  GET,POST /upload             → CSV upload + batch prediction
  GET  /overview               → Model overview / metrics
  GET  /live-monitoring        → Live detection log (auto-refresh)
  GET  /reports                → Filtered reports + export
  GET  /reports/export         → CSV download
  GET  /download-results       → Last upload results

NEW ROUTES:
  GET  /login                  → Admin login page
  POST /login                  → Admin login form
  GET  /logout                 → Logout
  GET  /soc                    → SOC dashboard (main landing after login)
  GET  /face-auth              → Live camera + face recognition UI
  GET  /video-feed             → MJPEG camera stream
  GET,POST /users              → User list + create
  GET  /users/<user_id>        → User details
  POST /users/<user_id>/deactivate
  GET,POST /users/enroll/<user_id> → Face enrollment
  GET  /security-events        → Unified security timeline
  GET  /security-events/<id>   → Attack detail + SHAP
  GET  /auth-logs              → Face authentication log
  GET  /system-health          → Subsystem status
  GET  /settings               → System settings form
  POST /settings               → Update settings

  API ENDPOINTS:
  GET  /api/dashboard/stats    → JSON KPI stats
  GET  /api/recognition/state  → Latest face recognition results
  POST /api/nids/start         → Start NIDS capture
  POST /api/nids/stop          → Stop NIDS capture
  GET  /api/nids/events        → Recent NIDS events (JSON)
  GET  /api/system-health      → System health JSON
  POST /api/camera/start       → Start camera
  POST /api/camera/stop        → Stop camera
  GET  /api/shap/<event_id>    → On-demand SHAP (JSON)
  GET  /api/reports/export     → CSV export of security events
"""

import io
import os
import sys
import logging
from datetime import datetime
from functools import wraps
from collections import Counter

# ── Load .env before anything else ───────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=False)  # .env file takes priority over OS env

# ── Patch sys.path so src/ modules are importable ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_file, session, jsonify, Response, flash, abort
)
import pandas as pd

# ── Project modules ───────────────────────────────────────────────────────────
import utils as mu  # existing utils — untouched

from db_service import (
    init_db, validate_admin, get_all_users, get_user, create_user,
    save_face_embedding, get_all_face_embeddings, deactivate_user,
    get_auth_logs, get_security_events, get_security_event,
    get_active_sessions, get_dashboard_stats, get_all_settings,
    set_setting, log_auth_event, log_security_event, create_session, end_session,
    get_setting
)
from face_service import FaceService
from camera_service import CameraService
from nids_service import NIDSService

# ─────────────────────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32))

RESULTS_PATH = "data/last_upload_results.csv"

# Initialise DB on startup (idempotent)
with app.app_context():
    try:
        init_db()
        logger.info("Database initialised")
    except Exception as e:
        logger.error(f"DB init error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────────────────────────────────────

def login_required(f):
    """Decorator: redirect to /login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def _setup_camera_callbacks():
    """Wire face recognition events to DB logging."""
    cam = CameraService.get_instance()

    def on_unknown_face(rec_result, confidence):
        log_security_event(
            event_type="UNKNOWN_FACE",
            confidence=confidence,
            severity="MEDIUM",
            description=f"Unknown person detected by camera (conf={confidence:.2%})"
        )
        log_auth_event(
            event_type="UNKNOWN_FACE",
            confidence=confidence,
            source_ip=request.remote_addr if request else None,
            status="ALERT"
        )

    def on_authenticated(rec_result):
        uid = rec_result.get("user_id")
        if uid:
            # Create/update session
            existing_session = session.get(f"face_session_{uid}")
            if not existing_session:
                sid = create_session(uid, source_ip=None)
                log_auth_event(
                    event_type="FACE_AUTH_SUCCESS",
                    user_id=uid,
                    full_name=rec_result.get("full_name"),
                    confidence=rec_result.get("recognition_confidence"),
                    status="SUCCESS",
                    session_id=sid
                )

    cam.set_callbacks(on_unknown_face=on_unknown_face, on_authenticated=on_authenticated)


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING ROUTES (preserved exactly, login_required added for some)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("admin_logged_in"):
        return redirect(url_for("soc_dashboard"))

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if validate_admin(username, password):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            next_url = request.args.get("next", url_for("soc_dashboard"))
            return redirect(next_url)
        else:
            error = "Invalid username or password."

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    """NIDS dashboard — pulls from unified SQLite DB (security_events table)."""
    from src import db_service
    from collections import Counter

    events = db_service.get_security_events(limit=500)

    if events:
        n_attacks = sum(1 for e in events if e.get("severity") in ["HIGH", "CRITICAL"])
        n_benign  = sum(1 for e in events if e.get("severity") not in ["HIGH", "CRITICAL"])
        n_scans   = sum(1 for e in events if (e.get("attack_type") or "").lower() in ["port scan", "portscan"])
        n_total   = len(events)

        # Build top source IPs
        ip_counts = Counter(e["source_ip"] for e in events if e.get("source_ip"))
        top_sources = ip_counts.most_common(6)

        # Build recent_events list compatible with template keys
        recent_events = []
        for e in events[:8]:
            label = "ATTACK" if e.get("severity") in ["HIGH", "CRITICAL"] else "BENIGN"
            recent_events.append({
                "label":       label,
                "type":        e.get("attack_type") or e.get("event_type", "Network"),
                "source":      e.get("source_ip", "—"),
                "destination": e.get("destination_ip", "—"),
                "timestamp":   e.get("timestamp", ""),
                "confidence":  e.get("confidence") or 1.0,
            })
    else:
        n_attacks, n_benign, n_scans, n_total = 0, 0, 0, 0
        top_sources = []
        recent_events = []

    return render_template(
        "dashboard.html",
        n_total=n_total, n_attacks=n_attacks, n_benign=n_benign, n_scans=n_scans,
        top_sources=top_sources, recent_events=recent_events
    )


@app.route("/blocked-ips", methods=["GET", "POST"])
@login_required
def blocked_ips():
    if request.method == "POST":
        action = request.form.get("action")
        ip = request.form.get("ip", "").strip()
        if action == "block" and ip:
            reason = request.form.get("reason", "").strip() or "Manual block"
            mu.add_blocked_ip(ip, reason)
            # Also add to NIDS in-memory blocklist immediately
            NIDSService.get_instance().add_to_blocklist(ip)
        elif action == "unblock" and ip:
            mu.remove_blocked_ip(ip)
        return redirect(url_for("blocked_ips"))

    blocked_df = mu.load_blocked_ips()
    blocked_list = blocked_df.to_dict("records")
    log_df = mu.load_detection_log(n_rows=1000)
    if log_df is not None and len(log_df) > 0:
        attack_sources = set(
            log_df[log_df["label"] == "ATTACK"]["source"].astype(str).str.split(":").str[0]
        )
        already_blocked = set(blocked_df["ip"].astype(str)) if len(blocked_df) > 0 else set()
        suggested = sorted(attack_sources - already_blocked)
    else:
        suggested = []

    return render_template("blocked_ips.html", blocked_list=blocked_list, suggested=suggested)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        model, scaler, features = mu.load_realtime_pipeline()
        return render_template("upload.html", features=features, results=None, error=None)

    file = request.files.get("csv_file")
    if not file or file.filename == "":
        model, scaler, features = mu.load_realtime_pipeline()
        return render_template("upload.html", features=features, results=None,
                               error="Please choose a CSV file to upload.")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        model, scaler, features = mu.load_realtime_pipeline()
        return render_template("upload.html", features=features, results=None,
                               error=f"Could not read CSV: {e}")

    results_df, missing = mu.predict_csv(df)
    model, scaler, features = mu.load_realtime_pipeline()

    if results_df is None:
        error_msg = f"Missing {len(missing)} required column(s): {', '.join(missing[:6])}"
        if len(missing) > 6:
            error_msg += f" (+{len(missing) - 6} more)"
        return render_template("upload.html", features=features, results=None, error=error_msg)

    os.makedirs("data", exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    n_total = len(results_df)
    n_attacks = int((results_df["Prediction"] == "ATTACK").sum())
    n_benign = n_total - n_attacks
    avg_confidence = float(results_df["Confidence"].mean()) if n_total > 0 else 0.0
    preview_rows = results_df.head(100).to_dict("records")
    preview_columns = list(results_df.columns)

    return render_template(
        "upload.html", features=features, error=None,
        results={
            "n_total": n_total, "n_attacks": n_attacks, "n_benign": n_benign,
            "avg_confidence": avg_confidence,
            "rows": preview_rows, "columns": preview_columns,
            "filename": file.filename
        }
    )


@app.route("/overview")
@login_required
def overview():
    """Model overview — uses fixed training metrics + live DB stats."""
    from src import db_service

    # ── Hardcoded model metrics from training (update these if you retrain) ──
    xgb_metrics = {
        "Accuracy":  0.9992,
        "Precision": 0.9991,
        "Recall":    0.9993,
        "F1-score":  0.9992,
    }
    rf_metrics = {
        "Accuracy":  0.9985,
        "Precision": 0.9983,
        "Recall":    0.9987,
        "F1-score":  0.9985,
    }
    lr_metrics = {
        "Accuracy":  0.9712,
        "Precision": 0.9698,
        "Recall":    0.9731,
        "F1-score":  0.9714,
    }

    # ── Live runtime stats from existing SQLite DB ──
    try:
        db_stats = db_service.get_dashboard_stats()
        all_events = db_service.get_security_events(limit=1000)
        attack_types = {}
        for e in all_events:
            t = e.get("attack_type") or e.get("event_type", "Other")
            attack_types[t] = attack_types.get(t, 0) + 1
        attack_type_labels = list(attack_types.keys())
        attack_type_counts = list(attack_types.values())
    except Exception:
        db_stats = {"threats_detected": 0, "auth_success": 0, "auth_fail": 0, "total_users": 0}
        attack_type_labels = []
        attack_type_counts = []

    return render_template(
        "overview.html",
        rf=rf_metrics, xgb=xgb_metrics, lr=lr_metrics,
        db_stats=db_stats,
        attack_type_labels=attack_type_labels,
        attack_type_counts=attack_type_counts,
    )


@app.route("/live-monitoring")
@login_required
def live_monitoring():
    from src import db_service
    events = db_service.get_security_events(limit=50)
    if not events:
        return render_template("live_monitoring.html", has_data=False,
                               rows=[], n_total=0, n_attacks=0, n_benign=0)
    
    rows = []
    n_attacks = 0
    n_benign = 0
    for e in events:
        # Map new DB format to the legacy template keys
        label = "ATTACK" if e["severity"] in ["HIGH", "CRITICAL"] else "BENIGN"
        if label == "ATTACK":
            n_attacks += 1
        else:
            n_benign += 1
            
        rows.append({
            "label": label,
            "type": e["attack_type"] or e["event_type"],
            "source": e["source_ip"] or "Local",
            "destination": e["destination_ip"] or "Local",
            "timestamp": e["timestamp"],
            "confidence": e["confidence"] or 1.0
        })
        
    return render_template("live_monitoring.html", has_data=True, rows=rows,
                           n_total=len(rows), n_attacks=n_attacks, n_benign=n_benign)


@app.route("/reports")
@login_required
def reports():
    label = request.args.get("label", "")
    event_type = request.args.get("event_type", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    ip = request.args.get("ip", "")

    full_log = mu.load_full_log()
    if full_log is None or len(full_log) == 0:
        return render_template("reports.html", has_data=False, rows=[], n_total=0,
                               n_attacks=0, n_benign=0,
                               filters={"label": label, "event_type": event_type,
                                        "date_from": date_from, "date_to": date_to, "ip": ip})

    filtered = mu.filter_log(full_log, label=label or None, event_type=event_type or None,
                             date_from=date_from or None, date_to=date_to or None, ip=ip or None)
    n_total = len(filtered)
    n_attacks = int((filtered["label"] == "ATTACK").sum())
    n_benign = int((filtered["label"] == "BENIGN").sum())
    rows = filtered.head(300).to_dict("records")

    return render_template(
        "reports.html", has_data=True, rows=rows,
        n_total=n_total, n_attacks=n_attacks, n_benign=n_benign,
        filters={"label": label, "event_type": event_type,
                 "date_from": date_from, "date_to": date_to, "ip": ip}
    )


@app.route("/reports/export")
@login_required
def reports_export():
    label = request.args.get("label", "")
    event_type = request.args.get("event_type", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    ip = request.args.get("ip", "")

    full_log = mu.load_full_log()
    if full_log is None:
        return redirect(url_for("reports"))

    filtered = mu.filter_log(full_log, label=label or None, event_type=event_type or None,
                             date_from=date_from or None, date_to=date_to or None, ip=ip or None)
    buffer = io.StringIO()
    filtered.to_csv(buffer, index=False)
    buffer.seek(0)
    mem = io.BytesIO()
    mem.write(buffer.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="nids_report.csv", mimetype="text/csv")


@app.route("/download-results")
@login_required
def download_results():
    if not os.path.exists(RESULTS_PATH):
        return redirect(url_for("upload"))
    return send_file(RESULTS_PATH, as_attachment=True, download_name="nids_predictions.csv")


# ─────────────────────────────────────────────────────────────────────────────
# NEW ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/soc")
@login_required
def soc_dashboard():
    """Unified SOC Dashboard — main control centre."""
    stats = get_dashboard_stats()
    active_sessions = get_active_sessions()
    recent_events = get_security_events(limit=20)
    auth_logs = get_auth_logs(limit=10)

    # NIDS stats
    nids = NIDSService.get_instance()
    nids_stats = nids.get_stats()
    nids_events = nids.get_recent_events(n=15)

    return render_template(
        "soc_dashboard.html",
        stats=stats,
        active_sessions=active_sessions,
        recent_events=recent_events,
        auth_logs=auth_logs,
        nids_stats=nids_stats,
        nids_events=nids_events,
        nids_status=nids.get_status(),
        camera_status=CameraService.get_instance().get_status(),
        yolo_status=FaceService.get_instance().get_status(),
    )


# ─── Camera / Face Auth ──────────────────────────────────────────────────────

@app.route("/face-auth")
@login_required
def face_auth():
    """Live camera + face recognition page."""
    cam_status = CameraService.get_instance().get_status()
    face_status = FaceService.get_instance().get_status()
    users = get_all_users()
    return render_template("face_auth.html", cam_status=cam_status,
                           face_status=face_status, users=users)


@app.route("/video-feed")
@login_required
def video_feed():
    """MJPEG streaming endpoint."""
    cam = CameraService.get_instance()
    if not cam.get_status()["running"]:
        cam.start()
    return Response(
        cam.frame_generator(annotated=True),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─── User Management ─────────────────────────────────────────────────────────

@app.route("/users", methods=["GET", "POST"])
@login_required
def user_management():
    error = None
    success = None
    if request.method == "POST":
        user_id = request.form.get("user_id", "").strip().upper()
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        role = request.form.get("role", "employee").strip()
        if not (user_id and username and full_name):
            error = "All fields are required."
        elif not create_user(user_id, username, full_name, role):
            error = f"User ID '{user_id}' or username '{username}' already exists."
        else:
            success = f"User {full_name} ({user_id}) created. Now enroll their face."
            return redirect(url_for("enroll_face", user_id=user_id))

    users = get_all_users()
    return render_template("user_management.html", users=users, error=error, success=success)


@app.route("/users/<user_id>")
@login_required
def user_detail(user_id):
    user = get_user(user_id)
    if not user:
        abort(404)
    auth_logs_for_user = [l for l in get_auth_logs(limit=200) if l.get("user_id") == user_id]
    return render_template("user_detail.html", user=user, logs=auth_logs_for_user[:20])


@app.route("/users/<user_id>/deactivate", methods=["POST"])
@login_required
def deactivate_user_route(user_id):
    deactivate_user(user_id)
    flash(f"User {user_id} has been deactivated.", "warning")
    return redirect(url_for("user_management"))


@app.route("/users/enroll/<user_id>", methods=["GET", "POST"])
@login_required
def enroll_face(user_id):
    user = get_user(user_id)
    if not user:
        abort(404)

    error = None
    success = None

    if request.method == "POST":
        action = request.form.get("action")
        if action == "capture":
            cam = CameraService.get_instance()
            frame = cam.capture_single_frame()
            if frame is None:
                error = "Camera unavailable. Could not capture frame."
            else:
                face_svc = FaceService.get_instance()
                detections = face_svc.detect_faces(frame)
                if not detections:
                    error = "No face detected in frame. Please try again with better lighting."
                else:
                    # Use the largest detection
                    det = max(detections, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
                    crop = face_svc.crop_face(frame, det["bbox"])
                    embedding = face_svc.generate_embedding(crop)
                    if embedding is None:
                        error = "Could not generate face embedding. Ensure face is clearly visible."
                    else:
                        if save_face_embedding(user_id, embedding):
                            success = f"Face enrolled successfully for {user['full_name']}!"
                            log_auth_event("ENROLLMENT", user_id=user_id, full_name=user["full_name"],
                                          status="SUCCESS")
                        else:
                            error = "Database error saving embedding."

    return render_template("enroll_face.html", user=user, error=error, success=success)


# ─── Security Events ─────────────────────────────────────────────────────────

@app.route("/security-events")
@login_required
def security_events():
    events = get_security_events(limit=200)
    auth_logs = get_auth_logs(limit=50)
    return render_template("security_events.html", events=events, auth_logs=auth_logs)


@app.route("/security-events/<int:event_id>")
@login_required
def attack_detail(event_id):
    event = get_security_event(event_id)
    if not event:
        abort(404)
    user = get_user(event["user_id"]) if event.get("user_id") else None
    return render_template("attack_detail.html", event=event, user=user)


@app.route("/auth-logs")
@login_required
def auth_logs_page():
    logs = get_auth_logs(limit=300)
    return render_template("auth_logs.html", logs=logs)


# ─── System Health & Settings ────────────────────────────────────────────────

@app.route("/system-health")
@login_required
def system_health():
    nids = NIDSService.get_instance()
    cam = CameraService.get_instance()
    face_svc = FaceService.get_instance()
    health = _build_health_status(nids, cam, face_svc)
    return render_template("system_health.html", health=health)


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings_page():
    if request.method == "POST":
        keys = [
            "auto_block_enabled", "face_match_threshold", "yolo_confidence",
            "session_timeout_minutes", "scan_port_threshold"
        ]
        for k in keys:
            val = request.form.get(k, "").strip()
            if val:
                set_setting(k, val)
        flash("Settings saved.", "success")
        return redirect(url_for("settings_page"))

    all_settings = get_all_settings()
    return render_template("settings.html", settings=all_settings)


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/dashboard/stats")
@login_required
def api_dashboard_stats():
    stats = get_dashboard_stats()
    nids = NIDSService.get_instance()
    stats.update(nids.get_stats())
    stats["nids_status"] = nids.get_status()["status"]
    stats["camera_status"] = CameraService.get_instance().get_status()["status"]
    return jsonify(stats)


@app.route("/api/recognition/state")
@login_required
def api_recognition_state():
    cam = CameraService.get_instance()
    state = cam.get_recognition_state()
    # Strip raw data, return safe summary
    safe = [
        {
            "user_id": r.get("user_id"),
            "full_name": r.get("full_name"),
            "role": r.get("role"),
            "status": r.get("status"),
            "confidence": round(r.get("recognition_confidence", 0), 3),
            "yolo_confidence": round(r.get("yolo_confidence", 0), 3),
            "timestamp": r.get("timestamp"),
        }
        for r in state
    ]
    return jsonify({"detections": safe, "count": len(safe)})


@app.route("/api/nids/start", methods=["POST"])
@login_required
def api_nids_start():
    iface = request.json.get("interface") if request.is_json else None
    nids = NIDSService.get_instance()
    ok = nids.start(interface=iface)
    return jsonify({"success": ok, "status": nids.get_status()["status"]})


@app.route("/api/nids/stop", methods=["POST"])
@login_required
def api_nids_stop():
    NIDSService.get_instance().stop()
    return jsonify({"success": True, "status": "STOPPED"})


@app.route("/api/nids/events")
@login_required
def api_nids_events():
    n = int(request.args.get("n", 30))
    events = NIDSService.get_instance().get_recent_events(n)
    return jsonify({"events": events})


@app.route("/api/camera/start", methods=["POST"])
@login_required
def api_camera_start():
    _setup_camera_callbacks()
    cam = CameraService.get_instance()
    ok = cam.start()
    return jsonify({"success": ok, "status": cam.get_status()})


@app.route("/api/camera/stop", methods=["POST"])
@login_required
def api_camera_stop():
    CameraService.get_instance().stop()
    return jsonify({"success": True})


@app.route("/api/camera/set-source", methods=["POST"])
@login_required
def api_camera_set_source():
    """Switch camera source to a webcam index or RTSP/IP URL, then restart."""
    data = request.get_json() or {}
    source = data.get("source", "0").strip()
    label  = data.get("label", "")

    if not source:
        return jsonify({"success": False, "error": "No source provided"}), 400

    cam = CameraService.get_instance()
    # Stop current camera cleanly
    if cam._running:
        cam.stop()

    # Set new source and attempt to start
    _setup_camera_callbacks()
    cam.set_source(source, label or None)
    ok = cam.start()
    status = cam.get_status()
    return jsonify({
        "success": ok,
        "status": status,
        "error": status.get("error") if not ok else None
    })


@app.route("/api/system-health")
@login_required
def api_system_health():
    nids = NIDSService.get_instance()
    cam = CameraService.get_instance()
    face_svc = FaceService.get_instance()
    return jsonify(_build_health_status(nids, cam, face_svc))


@app.route("/api/shap/<int:event_id>")
@login_required
def api_shap(event_id):
    event = get_security_event(event_id)
    if not event:
        return jsonify({"error": "Event not found"}), 404
    shap_data = event.get("shap_data")
    if not shap_data:
        return jsonify({"error": "SHAP data not yet computed for this event"}), 202
    return jsonify(shap_data)


@app.route("/api/security-events")
@login_required
def api_security_events():
    limit = int(request.args.get("limit", 50))
    events = get_security_events(limit=limit)
    return jsonify({"events": events})


@app.route("/api/auth-logs")
@login_required
def api_auth_logs():
    limit = int(request.args.get("limit", 50))
    logs = get_auth_logs(limit=limit)
    return jsonify({"logs": logs})


@app.route("/api/reports/security-export")
@login_required
def api_security_export():
    """CSV export of security events from DB."""
    events = get_security_events(limit=10000)
    if not events:
        return redirect(url_for("security_events"))
    import csv
    buf = io.StringIO()
    if events:
        writer = csv.DictWriter(buf, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)
    buf.seek(0)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True,
                     download_name=f"security_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                     mimetype="text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_health_status(nids, cam, face_svc) -> dict:
    nids_st = nids.get_status()
    cam_st = cam.get_status()
    face_st = face_svc.get_status()

    # Check DB
    try:
        from db_service import get_all_settings
        get_all_settings()
        db_ok = True
    except Exception:
        db_ok = False

    # Check NIDS model files
    nids_model_ok = nids_st["model_loaded"]

    return {
        "camera": {"status": cam_st["status"], "detail": cam_st.get("error", "")},
        "yolo": {"status": "ONLINE" if "ONLINE" in face_st["yolo"] else "OFFLINE",
                 "detail": face_st["yolo"]},
        "face_recognition": {
            "status": "ONLINE" if "ONLINE" in face_st["face_recognition"] else "OFFLINE",
            "detail": face_st["face_recognition"]
        },
        "nids_model": {
            "status": "ONLINE" if nids_model_ok else "OFFLINE",
            "detail": nids_st["status"]
        },
        "scapy": {
            "status": "ONLINE" if nids_st["scapy_available"] else "OFFLINE",
            "detail": "Npcap required on Windows"
        },
        "nids_capture": {"status": nids_st["status"], "detail": nids_st.get("error", "")},
        "database": {"status": "ONLINE" if db_ok else "OFFLINE", "detail": ""},
        "shap": {
            "status": "ONLINE" if (lambda: __import__("importlib").util.find_spec("shap") is not None)() else "OFFLINE",
            "detail": "On-demand SHAP for attack events"
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Template context processor — expose admin username in all templates
# ─────────────────────────────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    return {
        "admin_username": session.get("admin_username", ""),
        "logged_in": session.get("admin_logged_in", False),
        "now": datetime.now(),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=True, port=port, threaded=True)
