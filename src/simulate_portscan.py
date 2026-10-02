"""
simulate_portscan.py
----------------------
A simple attack simulator for TESTING your real-time detector.
It rapidly attempts connections to many ports on your own machine,
mimicking the pattern of a port scan (many quick connections, mostly
failing/refused, in a short time window).

Run this WHILE realtime_detection.py is running in another terminal.
This does NOT require Nmap or any extra installation - just Python's
built-in 'socket' library.

Usage:
    python src\\simulate_portscan.py
"""

import socket
import time

TARGET_IP = "127.0.0.1"   # <-- your current real local IP (check with 'ipconfig' if it changes again)
START_PORT = 1
END_PORT = 300               # scan more ports for a stronger burst pattern
TIMEOUT = 0.05                # much shorter timeout = faster, punchier scan

print(f"Simulating a port scan against {TARGET_IP} ")
print(f"Scanning ports {START_PORT} to {END_PORT}...")
print("Make sure realtime_detection.py is running in another terminal!\n")

start_time = time.time()

for port in range(START_PORT, END_PORT + 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    try:
        result = sock.connect_ex((TARGET_IP, port))
        if result == 0:
            print(f"  Port {port}: OPEN")
    except Exception:
        pass
    finally:
        sock.close()

duration = time.time() - start_time
print(f"\nScan complete. Scanned {END_PORT - START_PORT + 1} ports in {duration:.2f} seconds.")
print("Check the realtime_detection.py terminal for ATTACK alerts.")