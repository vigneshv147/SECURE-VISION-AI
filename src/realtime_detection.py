import time
import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from scapy.all import sniff, IP, TCP, UDP
 
# --- Log file: every detection result gets appended here, so the
# Streamlit dashboard can display a live feed of real detections ---
LOG_FILE = "logs/detection_log.csv"
os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "type", "source", "destination", "label", "confidence"])
 
 
def log_event(event_type, source, destination, label, confidence):
    """Append one detection event to the shared log file."""
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type, source, destination, label, f"{confidence:.4f}"
        ])
 
# --- OPTIONAL: set this to capture loopback traffic (127.0.0.1) on Windows ---
# Leave as None to capture on your normal default network adapter (Wi-Fi/Ethernet).
# Set to the Npcap loopback adapter name if you want to test with 127.0.0.1.
# To find the exact name, run this first:
#   python -c "from scapy.all import get_windows_if_list; [print(i['name']) for i in get_windows_if_list()]"
CAPTURE_INTERFACE = None
 
# --- Load the real-time model, scaler, and feature list ---
model = joblib.load("models/realtime_model.pkl")
scaler = joblib.load("models/realtime_scaler.pkl")
selected_features = joblib.load("models/realtime_features.pkl")
 
print("Loaded real-time model and scaler.")
print("Watching for these features:", selected_features)
 
 
def load_blocklist():
    """Load known-bad IPs from the blocklist CSV managed by the dashboard."""
    blocklist_path = "data/blocked_ips.csv"
    if not os.path.exists(blocklist_path):
        return set()
    df = pd.read_csv(blocklist_path)
    return set(df["ip"].astype(str).tolist())
 
 
blocked_ips = load_blocklist()
print(f"Loaded {len(blocked_ips)} blocked IP(s) from blocklist.")
 
# --- Rate-based scan detection settings ---
# A port scan is defined by BEHAVIOR (many ports contacted quickly),
# not by any single flow's content - so we track this separately,
# alongside the ML model, rather than relying on the model alone.
SCAN_PORT_THRESHOLD = 20     # if a source IP touches more than this many distinct ports...
SCAN_TIME_WINDOW = 10        # ...within this many seconds, flag it as a scan
 
# Tracks: {source_ip: [(timestamp, dest_port), ...]}
connection_log = {}
 
 
def log_connection_attempt(src_ip, dest_port):
    """Record that src_ip attempted to contact dest_port, for scan-rate tracking."""
    now = time.time()
    if src_ip not in connection_log:
        connection_log[src_ip] = []
    connection_log[src_ip].append((now, dest_port))
 
    # Drop entries older than our time window to keep this memory-efficient
    cutoff = now - SCAN_TIME_WINDOW
    connection_log[src_ip] = [(t, p) for t, p in connection_log[src_ip] if t >= cutoff]
 
 
def check_for_port_scans():
    """Look at recent connection activity and flag any source IP that has
    contacted an unusually large number of distinct ports recently."""
    for src_ip, attempts in list(connection_log.items()):
        distinct_ports = set(p for t, p in attempts)
        if len(distinct_ports) > SCAN_PORT_THRESHOLD:
            print(f"[ATTACK - PORT SCAN] {src_ip} contacted {len(distinct_ports)} "
                  f"distinct ports in the last {SCAN_TIME_WINDOW} seconds!")
            log_event("PORT_SCAN", src_ip, "multiple ports", "ATTACK", 1.0)
            # Reset after alerting so we don't spam the same alert every loop
            connection_log[src_ip] = []
# Key: (src_ip, dst_ip, src_port, dst_port, protocol)
# Value: dict of accumulated packet data for that flow
flows = {}
 
# How long (seconds) to wait with no new packets before we consider a flow "finished"
FLOW_TIMEOUT = 5
 
 
def get_flow_key(pkt):
    """Build a 5-tuple key identifying this packet's flow, and figure out
    if this packet is going 'forward' (matches the flow's original direction)
    or 'backward' (a reply)."""
    ip_layer = pkt[IP]
    proto = ip_layer.proto  # 6 = TCP, 17 = UDP
 
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        return None, None
 
    forward_key = (ip_layer.src, ip_layer.dst, sport, dport, proto)
    backward_key = (ip_layer.dst, ip_layer.src, dport, sport, proto)
 
    if forward_key in flows:
        return forward_key, "forward"
    elif backward_key in flows:
        return backward_key, "backward"
    else:
        return forward_key, "forward"  # new flow, this packet defines "forward"
 
 
def process_packet(pkt):
    """Called automatically by Scapy for every captured packet."""
    if IP not in pkt:
        return  # skip non-IP packets (ARP, etc.)
 
    key, direction = get_flow_key(pkt)
    if key is None:
        return  # skip non-TCP/UDP packets
 
    now = time.time()
    pkt_len = len(pkt)
    ip_layer = pkt[IP]
 
    # Header length: IP header + TCP/UDP header (rough estimate)
    header_len = pkt[IP].ihl * 4
    if TCP in pkt:
        header_len += pkt[TCP].dataofs * 4
        flags = pkt[TCP].flags
        window = pkt[TCP].window
    else:
        header_len += 8  # UDP header is fixed 8 bytes
        flags = 0
        window = 0
 
    payload_len = pkt_len - header_len
 
    # --- Create a new flow record if this is the first packet we've seen ---
    if key not in flows:
        flows[key] = {
            "start_time": now,
            "last_time": now,
            "fwd_lengths": [], "bwd_lengths": [],
            "fwd_headers": 0, "bwd_headers": 0,
            "fwd_times": [], "bwd_times": [],
            "psh_count": 0, "ack_count": 0,
            "init_win_fwd": None, "init_win_bwd": None,
            "dest_port": pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
            "act_data_pkt_fwd": 0,
            "min_seg_size_fwd": None,
        }
        # A brand-new flow means a new connection attempt - log it for scan detection
        if direction == "forward":
            log_connection_attempt(ip_layer.src, flows[key]["dest_port"])
 
    flow = flows[key]
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
 
    # TCP flags: PSH = 0x08, ACK = 0x10
    if flags:
        if flags & 0x08:
            flow["psh_count"] += 1
        if flags & 0x10:
            flow["ack_count"] += 1
 
 
def compute_features(flow):
    """Turn one flow's raw accumulated data into the 25 feature values
    our model expects, in the correct order."""
 
    def safe_std(values):
        return float(np.std(values)) if len(values) > 1 else 0.0
 
    def safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0
 
    def safe_min(values):
        return float(np.min(values)) if len(values) > 0 else 0.0
 
    def safe_max(values):
        return float(np.max(values)) if len(values) > 0 else 0.0
 
    duration = max(flow["last_time"] - flow["start_time"], 1e-6)  # avoid divide by zero
 
    all_times = sorted(flow["fwd_times"] + flow["bwd_times"])
    iat_all = np.diff(all_times) if len(all_times) > 1 else np.array([0.0])
    iat_fwd = np.diff(sorted(flow["fwd_times"])) if len(flow["fwd_times"]) > 1 else np.array([0.0])
 
    total_bytes = sum(flow["fwd_lengths"]) + sum(flow["bwd_lengths"])
 
    feature_values = {
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
        "Idle Min": 0.0,  # simplified - would need burst/idle period tracking
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
 
    return feature_values
 
 
def check_finished_flows():
    """Look for flows that have been idle longer than FLOW_TIMEOUT,
    classify them, print the result, and remove them from memory."""
    now = time.time()
    finished_keys = [k for k, f in flows.items() if now - f["last_time"] > FLOW_TIMEOUT]
 
    for key in finished_keys:
        flow = flows[key]
        src_ip, dst_ip, sport, dport, proto = key
 
        # --- Blocklist check: known-bad IPs are flagged immediately,
        # without needing the ML model to re-evaluate them every time ---
        if src_ip in blocked_ips:
            print(f"[BLOCKED] {src_ip}:{sport} -> {dst_ip}:{dport}  "
                  f"(source IP is on the blocklist)")
            log_event("BLOCKED_IP", f"{src_ip}:{sport}", f"{dst_ip}:{dport}", "ATTACK", 1.0)
            del flows[key]
            continue
 
        features = compute_features(flow)
 
        # Build the feature vector in the EXACT order the model expects.
        # Using a DataFrame with matching column names avoids the sklearn
        # "does not have valid feature names" warning.
        vector = pd.DataFrame([[features[f] for f in selected_features]], columns=selected_features)
        vector_scaled = scaler.transform(vector)
 
        prediction = model.predict(vector_scaled)[0]
        probability = model.predict_proba(vector_scaled)[0][1]  # probability of ATTACK
 
        label = "ATTACK" if prediction == 1 else "BENIGN"
 
        print(f"[{label}] {src_ip}:{sport} -> {dst_ip}:{dport}  "
              f"(confidence: {probability:.2%})")
 
        log_event("FLOW", f"{src_ip}:{sport}", f"{dst_ip}:{dport}", label, probability)
 
        del flows[key]
 
 
def main():
    global blocked_ips
    print("\nStarting real-time detection... Press Ctrl+C to stop.\n")
    loop_count = 0
    try:
        while True:
            # Capture packets for 3 seconds at a time, then check for finished flows
            sniff(prn=process_packet, timeout=3, store=False, iface=CAPTURE_INTERFACE)
            check_finished_flows()
            check_for_port_scans()
 
            # Reload the blocklist every ~30 seconds so IPs blocked via the
            # dashboard take effect here without needing a restart
            loop_count += 1
            if loop_count % 10 == 0:
                blocked_ips = load_blocklist()
    except KeyboardInterrupt:
        print("\nStopping real-time detection.")
 
 
if __name__ == "__main__":
    main()
 