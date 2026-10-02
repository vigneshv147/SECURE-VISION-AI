"""
nids_service.py
----------------
Flask-integrated wrapper around the existing NIDS realtime detection.

IMPORTANT: This does NOT modify realtime_detection.py.
It re-implements the same logic as a thread-safe service that can be
controlled from the Flask app (start/stop/status) and emits events to
the database when attacks are detected.

The existing detection_log.csv format is PRESERVED for backward compatibility.
New attack events are ALSO written to the SQLite database for the unified dashboard.

PRESERVED from realtime_detection.py:
- Flow tracking (5-tuple)
- Feature extraction (25 features, exact same names/order)
- Blocklist check
- Port scan detection (rate-based)
- Scaler + model loading
"""

import os
import csv
import time
import logging
import threading
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Set, List
from collections import deque

logger = logging.getLogger(__name__)

# Scapy import (may fail if not admin / Npcap not installed)
try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("scapy not available — live packet capture disabled")

# SHAP (loaded lazily, only for attack events)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ──── Constants (match realtime_detection.py) ────────────────────────────────
LOG_FILE = "logs/detection_log.csv"
SCAN_PORT_THRESHOLD = int(os.environ.get("SCAN_PORT_THRESHOLD", "20"))
SCAN_TIME_WINDOW = int(os.environ.get("SCAN_TIME_WINDOW", "10"))
FLOW_TIMEOUT = int(os.environ.get("FLOW_TIMEOUT", "5"))


class NIDSService:
    """Thread-safe NIDS service that wraps the existing packet capture logic."""

    _instance: Optional["NIDSService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features: List[str] = []
        self.blocked_ips: Set[str] = set()
        self.flows: Dict = {}
        self.connection_log: Dict = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._status = "STOPPED"
        self._error: Optional[str] = None
        self._capture_interface = None
        self._recent_events = deque(maxlen=200)  # in-memory ring buffer for fast API
        self._shap_explainer = None

        # Stats counters
        self._stats = {
            "total_flows": 0,
            "attack_count": 0,
            "benign_count": 0,
            "blocked_count": 0,
            "scan_count": 0,
        }

        # Ensure log file exists
        os.makedirs("logs", exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "type", "source", "destination", "label", "confidence"]
                )

        self._load_models()

    @classmethod
    def get_instance(cls) -> "NIDSService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ─── Model Loading ───────────────────────────────────────────────────────

    def _load_models(self):
        model_path = os.environ.get("NIDS_MODEL_PATH", "models/realtime_model.pkl")
        scaler_path = os.environ.get("NIDS_SCALER_PATH", "models/realtime_scaler.pkl")
        feat_path = os.environ.get("NIDS_FEATURES_PATH", "models/realtime_features.pkl")

        missing = [p for p in [model_path, scaler_path, feat_path] if not os.path.exists(p)]
        if missing:
            self._status = f"OFFLINE — missing: {', '.join(missing)}"
            logger.error(self._status)
            return

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.selected_features = list(joblib.load(feat_path))
            self._status = "READY"
            logger.info(f"NIDS model loaded: {len(self.selected_features)} features")
        except Exception as e:
            self._status = f"OFFLINE — load error: {e}"
            logger.error(self._status)

    def _load_shap_explainer(self):
        """Lazily load SHAP TreeExplainer (expensive, done once on first attack)."""
        if self._shap_explainer is not None:
            return
        if not SHAP_AVAILABLE or self.model is None:
            return
        try:
            self._shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer loaded")
        except Exception as e:
            logger.error(f"SHAP load error: {e}")

    # ─── Start / Stop ────────────────────────────────────────────────────────

    def start(self, interface=None) -> bool:
        if self._running:
            return True
        if not SCAPY_AVAILABLE:
            self._status = "OFFLINE — scapy not available (install Npcap on Windows)"
            return False
        if self.model is None:
            self._status = "OFFLINE — NIDS model not loaded"
            return False

        self._capture_interface = interface
        self._running = True
        self._status = "ONLINE"
        self._reload_blocklist()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="NIDSThread"
        )
        self._thread.start()
        logger.info("NIDS packet capture started")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._status = "STOPPED"
        logger.info("NIDS stopped")

    # ─── Blocklist ───────────────────────────────────────────────────────────

    def _reload_blocklist(self):
        path = "data/blocked_ips.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            self.blocked_ips = set(df["ip"].astype(str).tolist())
        else:
            self.blocked_ips = set()

    def add_to_blocklist(self, ip: str):
        self.blocked_ips.add(ip)

    # ─── Capture Loop ────────────────────────────────────────────────────────

    def _capture_loop(self):
        loop_count = 0
        while self._running:
            try:
                sniff(
                    prn=self._process_packet,
                    timeout=3,
                    store=False,
                    iface=self._capture_interface,
                )
            except Exception as e:
                logger.error(f"Sniff error: {e}")
                time.sleep(1)

            self._check_finished_flows()
            self._check_port_scans()

            loop_count += 1
            if loop_count % 10 == 0:
                self._reload_blocklist()

    # ─── Packet Processing (identical logic to realtime_detection.py) ────────

    def _process_packet(self, pkt):
        if IP not in pkt:
            return
        key, direction = self._get_flow_key(pkt)
        if key is None:
            return

        now = time.time()
        pkt_len = len(pkt)
        ip_layer = pkt[IP]
        header_len = pkt[IP].ihl * 4

        if TCP in pkt:
            header_len += pkt[TCP].dataofs * 4
            flags = pkt[TCP].flags
            window = pkt[TCP].window
        else:
            header_len += 8
            flags = 0
            window = 0

        payload_len = pkt_len - header_len

        if key not in self.flows:
            self.flows[key] = {
                "start_time": now, "last_time": now,
                "fwd_lengths": [], "bwd_lengths": [],
                "fwd_headers": 0, "bwd_headers": 0,
                "fwd_times": [], "bwd_times": [],
                "psh_count": 0, "ack_count": 0,
                "init_win_fwd": None, "init_win_bwd": None,
                "dest_port": pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
                "act_data_pkt_fwd": 0,
                "min_seg_size_fwd": None,
            }
            if direction == "forward":
                self._log_connection(ip_layer.src, self.flows[key]["dest_port"])

        flow = self.flows[key]
        flow["last_time"] = now

        if direction == "forward":
            flow["fwd_lengths"].append(payload_len)
            flow["fwd_headers"] += header_len
            flow["fwd_times"].append(now)
            if payload_len > 0:
                flow["act_data_pkt_fwd"] += 1
            if flow["init_win_fwd"] is None:
                flow["init_win_fwd"] = window
            if flow["min_seg_size_fwd"] is None or header_len < flow["min_seg_size_fwd"]:
                flow["min_seg_size_fwd"] = header_len
        else:
            flow["bwd_lengths"].append(payload_len)
            flow["bwd_headers"] += header_len
            flow["bwd_times"].append(now)
            if flow["init_win_bwd"] is None:
                flow["init_win_bwd"] = window

        if flags:
            if flags & 0x08:
                flow["psh_count"] += 1
            if flags & 0x10:
                flow["ack_count"] += 1

    def _get_flow_key(self, pkt):
        ip_layer = pkt[IP]
        proto = ip_layer.proto
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        else:
            return None, None

        fwd = (ip_layer.src, ip_layer.dst, sport, dport, proto)
        bwd = (ip_layer.dst, ip_layer.src, dport, sport, proto)
        if fwd in self.flows:
            return fwd, "forward"
        elif bwd in self.flows:
            return bwd, "backward"
        return fwd, "forward"

    def _log_connection(self, src_ip, dest_port):
        now = time.time()
        if src_ip not in self.connection_log:
            self.connection_log[src_ip] = []
        self.connection_log[src_ip].append((now, dest_port))
        cutoff = now - SCAN_TIME_WINDOW
        self.connection_log[src_ip] = [
            (t, p) for t, p in self.connection_log[src_ip] if t >= cutoff
        ]

    def _check_port_scans(self):
        for src_ip, attempts in list(self.connection_log.items()):
            distinct = set(p for t, p in attempts)
            if len(distinct) > SCAN_PORT_THRESHOLD:
                logger.warning(f"PORT SCAN: {src_ip} → {len(distinct)} ports")
                self._log_event("PORT_SCAN", src_ip, "multiple", "ATTACK", 1.0)
                self._stats["scan_count"] += 1
                self._stats["attack_count"] += 1
                self._emit_security_event(
                    "PORT_SCAN", src_ip, "multiple", 1.0, "HIGH",
                    f"Port scan: {len(distinct)} distinct ports in {SCAN_TIME_WINDOW}s"
                )
                self.connection_log[src_ip] = []

    def _check_finished_flows(self):
        now = time.time()
        done = [k for k, f in self.flows.items() if now - f["last_time"] > FLOW_TIMEOUT]

        for key in done:
            flow = self.flows[key]
            src_ip, dst_ip, sport, dport, proto = key

            if src_ip in self.blocked_ips:
                self._log_event("BLOCKED_IP", f"{src_ip}:{sport}", f"{dst_ip}:{dport}", "ATTACK", 1.0)
                self._stats["blocked_count"] += 1
                del self.flows[key]
                continue

            features = self._compute_features(flow)
            vector = pd.DataFrame([[features[f] for f in self.selected_features]], columns=self.selected_features)
            vector_scaled = self.scaler.transform(vector)
            pred = self.model.predict(vector_scaled)[0]
            prob = float(self.model.predict_proba(vector_scaled)[0][1])
            label = "ATTACK" if pred == 1 else "BENIGN"

            self._log_event("FLOW", f"{src_ip}:{sport}", f"{dst_ip}:{dport}", label, prob)
            self._stats["total_flows"] += 1

            if label == "ATTACK":
                self._stats["attack_count"] += 1
                severity = "CRITICAL" if prob > 0.95 else ("HIGH" if prob > 0.85 else "MEDIUM")
                event_id = self._emit_security_event(
                    "NETWORK_ATTACK", src_ip, dst_ip, prob, severity,
                    f"ML detected attack flow {src_ip}:{sport}→{dst_ip}:{dport}"
                )
                # Compute SHAP for high-severity attacks asynchronously
                if prob > 0.80:
                    threading.Thread(
                        target=self._compute_shap_async,
                        args=(event_id, vector_scaled, features),
                        daemon=True
                    ).start()
            else:
                self._stats["benign_count"] += 1

            del self.flows[key]

    def _compute_features(self, flow) -> Dict:
        """Exact copy of compute_features from realtime_detection.py."""
        def safe_std(v): return float(np.std(v)) if len(v) > 1 else 0.0
        def safe_mean(v): return float(np.mean(v)) if v else 0.0
        def safe_min(v): return float(np.min(v)) if v else 0.0
        def safe_max(v): return float(np.max(v)) if v else 0.0

        duration = max(flow["last_time"] - flow["start_time"], 1e-6)
        all_times = sorted(flow["fwd_times"] + flow["bwd_times"])
        iat_all = np.diff(all_times) if len(all_times) > 1 else np.array([0.0])
        iat_fwd = np.diff(sorted(flow["fwd_times"])) if len(flow["fwd_times"]) > 1 else np.array([0.0])
        total_bytes = sum(flow["fwd_lengths"]) + sum(flow["bwd_lengths"])

        return {
            "Bwd Packet Length Std": safe_std(flow["bwd_lengths"]),
            "Total Length of Bwd Packets": sum(flow["bwd_lengths"]),
            "Destination Port": flow["dest_port"],
            "Total Length of Fwd Packets": sum(flow["fwd_lengths"]),
            "Bwd Header Length": flow["bwd_headers"],
            "Fwd Header Length": flow["fwd_headers"],
            "act_data_pkt_fwd": flow["act_data_pkt_fwd"],
            "Fwd Packet Length Max": safe_max(flow["fwd_lengths"]),
            "PSH Flag Count": flow["psh_count"],
            "Init_Win_bytes_backward": flow["init_win_bwd"] or 0,
            "Fwd IAT Std": safe_std(iat_fwd),
            "Idle Min": 0.0,
            "Fwd Packet Length Mean": safe_mean(flow["fwd_lengths"]),
            "ACK Flag Count": flow["ack_count"],
            "Fwd IAT Min": safe_min(iat_fwd),
            "min_seg_size_forward": flow["min_seg_size_fwd"] or 0,
            "Bwd Packets/s": len(flow["bwd_lengths"]) / duration,
            "Fwd IAT Mean": safe_mean(iat_fwd),
            "Init_Win_bytes_forward": flow["init_win_fwd"] or 0,
            "Fwd IAT Total": float(np.sum(iat_fwd)),
            "Bwd Packet Length Min": safe_min(flow["bwd_lengths"]),
            "Flow Bytes/s": total_bytes / duration,
            "Min Packet Length": safe_min(flow["fwd_lengths"] + flow["bwd_lengths"]),
            "Fwd Packet Length Min": safe_min(flow["fwd_lengths"]),
            "Flow IAT Min": safe_min(iat_all),
        }

    def _log_event(self, event_type, source, destination, label, confidence):
        """Append to CSV log (preserves existing format exactly)."""
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event_type, source, destination, label, f"{confidence:.4f}"
            ])
        # Also keep in memory ring buffer
        self._recent_events.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type, "source": source, "destination": destination,
            "label": label, "confidence": round(confidence, 4)
        })

    def _emit_security_event(self, event_type, src_ip, dst_ip, confidence, severity, description) -> int:
        """Write to DB and try to correlate with an authenticated session."""
        try:
            from db_service import log_security_event, get_session_by_ip
            session = get_session_by_ip(src_ip)
            user_id = session["user_id"] if session else None
            session_id = session["session_id"] if session else None
            event_id = log_security_event(
                event_type=event_type, source_ip=src_ip, destination_ip=dst_ip,
                attack_type=event_type, confidence=confidence, severity=severity,
                description=description, user_id=user_id, session_id=session_id
            )
            return event_id
        except Exception as e:
            logger.error(f"DB event write error: {e}")
            return -1

    def _compute_shap_async(self, event_id: int, vector_scaled, features: Dict):
        """Compute SHAP values in background thread and update DB event."""
        if event_id < 0:
            return
        try:
            import shap
            from db_service import update_event_shap
            self._load_shap_explainer()
            if self._shap_explainer is None:
                return
            sv = self._shap_explainer.shap_values(vector_scaled)
            # sv shape: (1, n_features) — squeeze
            if hasattr(sv, "shape") and len(sv.shape) == 2:
                sv = sv[0]
            elif isinstance(sv, list):
                sv = sv[1][0] if len(sv) > 1 else sv[0][0]

            feature_names = self.selected_features
            top_features = sorted(
                zip(feature_names, vector_scaled[0], sv),
                key=lambda x: abs(x[2]), reverse=True
            )[:10]

            shap_data = {
                "features": [
                    {"name": name, "value": round(float(val), 4), "contribution": round(float(contrib), 4)}
                    for name, val, contrib in top_features
                ],
                "base_value": round(float(self._shap_explainer.expected_value), 4) if hasattr(self._shap_explainer, "expected_value") else None
            }
            update_event_shap(event_id, shap_data)
            logger.info(f"SHAP computed for event {event_id}")
        except Exception as e:
            logger.error(f"SHAP computation error: {e}")

    # ─── Public API ──────────────────────────────────────────────────────────

    def get_recent_events(self, n: int = 50) -> List[Dict]:
        return list(self._recent_events)[-n:][::-1]

    def get_stats(self) -> Dict:
        return dict(self._stats)

    def get_status(self) -> Dict:
        return {
            "status": self._status,
            "error": self._error,
            "scapy_available": SCAPY_AVAILABLE,
            "model_loaded": self.model is not None,
            "features_count": len(self.selected_features),
            "active_flows": len(self.flows),
            "blocked_ips_loaded": len(self.blocked_ips),
        }

    def predict_flow_vector(self, feature_dict: Dict) -> Dict:
        """On-demand single flow prediction (for API use)."""
        if self.model is None:
            return {"error": "model not loaded"}
        try:
            vector = pd.DataFrame(
                [[feature_dict.get(f, 0) for f in self.selected_features]],
                columns=self.selected_features
            )
            vector_scaled = self.scaler.transform(vector)
            pred = self.model.predict(vector_scaled)[0]
            prob = float(self.model.predict_proba(vector_scaled)[0][1])
            return {"label": "ATTACK" if pred == 1 else "BENIGN", "confidence": prob}
        except Exception as e:
            return {"error": str(e)}
