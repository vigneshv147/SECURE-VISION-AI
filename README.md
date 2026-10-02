# NIDS-ML + AI SOC Platform 🛡️

A comprehensive Artificial Intelligence Security Operations Center (AI-SOC) integrating **Face Recognition Identity Management** with a real-time **Network Intrusion Detection System (NIDS)**.

This project unifies physical security (who is at the terminal) with network security (what traffic is being generated), enabling powerful correlation between authenticated user sessions and malicious network activity.

---

## 🌟 Key Features

### 1. Advanced Threat Detection (NIDS)
- **Machine Learning**: Real-time packet capture and flow classification using a trained XGBoost model (99.9% accuracy).
- **Behavioral Analysis**: Detects port scans using rate-based thresholds (e.g. 20 distinct ports within 10 seconds).
- **Explainable AI (XAI)**: Generates on-demand **SHAP (SHapley Additive exPlanations)** charts to explain *why* a network flow was flagged as an attack.
- **Automated Response**: Configurable Safe Mode (log only) or Active Blocking mode via dynamic blocklists.

### 2. Physical Identity & Access (Face Auth)
- **YOLO Face Detection**: Utilizes a YOLOv11 model to instantly detect known faces in the camera feed.
- **Biometric Enrolment**: Enrol users directly from the browser; generates and stores secure 128-dimensional face embeddings (powered by dlib/face_recognition).
- **Session Management**: Successful face authentication spawns a secure session tied to the local source IP. Unknown faces trigger Security Alerts.

### 3. User-Network Correlation (The SOC)
- **Correlated Security Events**: When a network attack occurs (e.g. a Port Scan), the system checks if the source IP belongs to an active, face-authenticated user session. If so, the attack is attributed directly to that identity.
- **Unified Timeline**: View all Identity events (Face Enrolled, Auth Success, Unknown Face) and Network events (Attack, Port Scan) in one seamless timeline.
- **Premium Dashboard**: A rich, dark-mode, glassmorphism UI built with Bootstrap 5.

---

## 🏗️ Architecture

```text
Camera Service → YOLO Detection → Face Embeddings → SQLite User Database
                                                         ↓ (Session Auth)
Network Interface → Scapy Capture → 25-Feature Extractor → XGBoost
                                                         ↓ (Attack Alert)
                                                    SHAP Explainer
                                                         ↓
                 [ UNIFIED AI SECURITY DASHBOARD ]
```

---

## 🚀 Installation & Setup (Windows)

### 1. Prerequisites
- Python 3.9 - 3.11
- **Npcap**: You MUST install Npcap for Scapy to capture live packets on Windows. Download from [nmap.org/npcap](https://nmap.org/npcap/) (Make sure to check "Install Npcap in WinPcap API-compatible Mode" during installation).
- **C++ Build Tools**: Required for compiling `dlib` (used by face-recognition). Install Visual Studio Build Tools with the "Desktop development with C++" workload.

### 2. Environment Setup

```powershell
# Clone the repository (or navigate to the directory)
cd AI-Based-Intrusion-Detection-System-main

# Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Copy the `.env.example` file to a new file named `.env`:
```powershell
cp .env.example .env
```
Edit `.env` to configure your admin credentials, camera index, thresholds, and paths.

### 4. Train the YOLO Model (Crucial Step)
The repository includes a YOLO training dataset (`AICS.v1i.yolov11/`) but **does not** include a pre-trained `.pt` file. You must train it:

```powershell
# Train a nano model on the provided dataset for 50 epochs
yolo train model=yolo11n.pt data=AICS.v1i.yolov11/data.yaml epochs=50 imgsz=704

# After training finishes, copy the best model to the required directory
mkdir -p models/yolo
cp runs/detect/train/weights/best.pt models/yolo/best.pt
```

*(Note: If you already have a trained YOLO model, simply place the `.pt` file at `models/yolo/best.pt` and ensure `YOLO_DATA_PATH` points to the `data.yaml` used to train it).*

---

## 💻 Running the Application

Start the unified Flask server:

```powershell
# Must be run as Administrator for packet capture to work!
python app.py
```

### Usage Workflow
1. Open a browser to `http://127.0.0.1:5000`
2. **Log In**: Use the admin credentials defined in your `.env` file.
3. **Enroll Users**: Navigate to `Identity > Users` and create an employee. Click the Camera icon to enroll their face.
4. **Start Camera**: Navigate to `Identity > Face Auth` and click **Activate Camera**. The system will begin continuously recognizing enrolled users.
5. **Start Network Monitoring**: Navigate to `Core > SOC Dashboard` and click **Start Network Monitor**.
6. **Simulate an Attack**: Open a second terminal and run `python src/simulate_portscan.py`.
7. **View Correlation**: Check the SOC Dashboard or Security Events tab to see the attack correlated with the face-authenticated user!

---

## 📂 Project Structure

- `app.py`: The main Flask server unifying all routes.
- `src/camera_service.py`: Persistent threaded camera loop.
- `src/face_service.py`: YOLO inference and face recognition/embedding logic.
- `src/nids_service.py`: Threaded packet capture and XGBoost inference.
- `src/db_service.py`: SQLite operations for users, sessions, and events.
- `src/realtime_detection.py`: Original legacy NIDS script (preserved).
- `models/`: Trained XGBoost and Random Forest `.pkl` artifacts.
- `templates/`: HTML Jinja2 templates (SOC dashboard, Face Auth, etc.).
- `static/`: CSS and JavaScript files (`soc.css` for premium styling).

---

## 🔒 Security & Privacy Notes

- **Biometrics**: Face embeddings are converted into non-reversible mathematical vectors (128-dim JSON arrays) and stored securely in the SQLite database. Raw face images are NOT stored on disk after enrollment.
- **Firewall Modifications**: The application operates in **Log-Only / Safe Mode** by default (`AUTO_BLOCK_ENABLED=false`). It will not automatically modify Windows firewall rules unless explicitly reconfigured by an administrator.

## 🐛 Troubleshooting

- **"Scapy Offline" in System Health**: You did not install Npcap, or you did not run `app.py` as an Administrator.
- **"YOLO Offline"**: Ensure you have trained the model and placed it at `models/yolo/best.pt`, or verify the path in your `.env` file.
- **Camera Not Opening**: Try changing `CAMERA_INDEX` in the `.env` file from `0` to `1` or `2`.
- **Dlib installation fails**: Install CMake (`pip install cmake`) and ensure Visual Studio C++ Build Tools are installed.
