"""
camera_service.py
------------------
Persistent camera loop with YOLO detection and face recognition.

DESIGN:
- Single background thread owns the camera resource (never re-opened per frame)
- Frames are processed at configurable intervals (FRAME_INTERVAL)
- MJPEG stream available via generator for Flask video feed
- Recognition results are stored in shared state dict for dashboard polling
- Unknown face events are logged to DB (not classified as malicious)

IMPORTANT for Windows:
- cv2.VideoCapture(0) should work with standard webcam
- If camera index 0 fails, try 1 or 2
- DirectShow backend: cv2.VideoCapture(0, cv2.CAP_DSHOW)
"""

import os
import cv2
import time
import threading
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports from services
from face_service import FaceService


class CameraService:
    """
    Manages a single webcam with threaded frame capture and YOLO+face recognition.
    """

    _instance: Optional["CameraService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.camera_index = int(os.environ.get("CAMERA_INDEX", "0"))
        self.camera_source = None          # None = use camera_index, str = RTSP/IP URL
        self.camera_label  = "Webcam"      # human-readable label shown in UI
        self.frame_interval = int(os.environ.get("FRAME_INTERVAL", "2"))
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._latest_annotated: Optional[bytes] = None
        self._recognition_state: List[Dict] = []
        self._frame_count = 0
        self._status = "STOPPED"
        self._error: Optional[str] = None

        # Callbacks for external event handling
        self._on_unknown_face = None
        self._on_authenticated = None

    @classmethod
    def get_instance(cls) -> "CameraService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set_callbacks(self, on_unknown_face=None, on_authenticated=None):
        self._on_unknown_face = on_unknown_face
        self._on_authenticated = on_authenticated

    def set_source(self, source: str, label: str = None):
        """
        Change camera source. Call before start(), or stop()+set_source()+start().
        source: '0','1','2' for webcam index, or RTSP/HTTP URL string.
        """
        if source.strip().isdigit():
            self.camera_index = int(source.strip())
            self.camera_source = None
            self.camera_label = f"Webcam (index {self.camera_index})"
        else:
            self.camera_source = source.strip()
            self.camera_label = label or "IP / Industrial Camera"
        logger.info(f"Camera source set to: {self.camera_label}")

    # ─── Start / Stop ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the camera loop. Returns True if camera opened successfully."""
        if self._running:
            return True

        try:
            # Determine source: RTSP URL or local index
            if self.camera_source:
                # Industrial / IP camera via RTSP or HTTP URL
                logger.info(f"Opening IP camera: {self.camera_source}")
                cap = cv2.VideoCapture(self.camera_source)
            else:
                # Local webcam — try DirectShow first (Windows)
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                src_desc = self.camera_source or f"index {self.camera_index}"
                self._status = "OFFLINE"
                self._error = f"Cannot open camera: {src_desc}"
                logger.error(self._error)
                return False

            # Apply resolution only for local webcams (RTSP streams ignore these)
            if not self.camera_source:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)

            # For RTSP: set buffer to 1 to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap = cap
        except Exception as e:
            self._status = "OFFLINE"
            self._error = str(e)
            logger.error(f"Camera open error: {e}")
            return False

        self._running = True
        self._status = "ONLINE"
        self._error = None
        self._frame_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        src_info = self.camera_source or f"index={self.camera_index}"
        logger.info(f"Camera started ({src_info})")
        return True

    def stop(self):
        """Stop the camera gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._status = "STOPPED"
        self._recognition_state = []
        logger.info("Camera stopped")

    # ─── Capture Loop ────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Background thread: reads frames, runs detection every N frames."""
        face_svc = FaceService.get_instance()
        known_embeddings = {}
        embed_refresh_counter = 0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._status = "OFFLINE"
                self._error = "Camera disconnected"
                break

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            self._frame_count += 1

            # Encode raw frame as JPEG for stream
            raw_jpeg = self._encode_jpeg(frame)
            with self._frame_lock:
                self._latest_frame = raw_jpeg

            # Run detection every FRAME_INTERVAL frames
            if self._frame_count % self.frame_interval == 0:
                # Refresh known embeddings every ~5 seconds
                embed_refresh_counter += 1
                if embed_refresh_counter % (30 // self.frame_interval) == 0:
                    try:
                        from db_service import get_all_face_embeddings
                        known_embeddings = get_all_face_embeddings()
                    except Exception:
                        pass

                detections = face_svc.detect_faces(frame)
                recognition_results = []

                for det in detections:
                    face_crop = face_svc.crop_face(frame, det["bbox"])
                    is_known_by_yolo = det.get("is_known_class", False)

                    # Default status:
                    # - If YOLO says "know" → start as DETECTED (person is known, just not yet matched)
                    # - If YOLO doesn't recognise class → UNKNOWN (intruder)
                    default_status = "DETECTED" if is_known_by_yolo else "UNKNOWN"

                    rec_result = {
                        "bbox": det["bbox"],
                        "yolo_class": det["class_name"],
                        "yolo_confidence": det["confidence"],
                        "user_id": None,
                        "full_name": None,
                        "role": None,
                        "recognition_confidence": 0.0,
                        "status": default_status,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Generate embedding and compare against enrolled users
                    if len(face_crop) > 0 and face_crop.size > 0:
                        embedding = face_svc.generate_embedding(face_crop)
                        if embedding and known_embeddings:
                            uid, sim = face_svc.compare_embedding(embedding, known_embeddings)
                            if uid:
                                # Full match — identity confirmed
                                user_data = known_embeddings[uid]
                                rec_result.update({
                                    "user_id": uid,
                                    "full_name": user_data["full_name"],
                                    "role": user_data["role"],
                                    "recognition_confidence": sim,
                                    "status": "AUTHENTICATED",
                                })
                                if self._on_authenticated:
                                    try:
                                        self._on_authenticated(rec_result)
                                    except Exception:
                                        pass
                            else:
                                # Embedding exists but no DB match:
                                # If YOLO said "know", keep as DETECTED (not a false alarm)
                                # Only fire unknown callback for non-known YOLO classes
                                if not is_known_by_yolo and self._on_unknown_face:
                                    try:
                                        self._on_unknown_face(rec_result, det["confidence"])
                                    except Exception:
                                        pass
                        # No embeddings in DB at all — status stays at default_status set above

                    recognition_results.append(rec_result)

                # Draw annotations
                annotated = face_svc.draw_detections(frame, detections, recognition_results)
                annotated_jpeg = self._encode_jpeg(annotated)

                with self._frame_lock:
                    self._latest_annotated = annotated_jpeg
                    self._recognition_state = recognition_results

            time.sleep(0.033)  # ~30 fps max capture

    def _encode_jpeg(self, frame: np.ndarray, quality: int = 85) -> bytes:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    # ─── Stream Generator ────────────────────────────────────────────────────

    def frame_generator(self, annotated: bool = True):
        """
        MJPEG generator for Flask streaming response.
        Usage: Response(camera.frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')
        """
        while self._running:
            with self._frame_lock:
                frame_bytes = self._latest_annotated if (annotated and self._latest_annotated) else self._latest_frame

            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.05)

    # ─── Public API ──────────────────────────────────────────────────────────

    def get_recognition_state(self) -> List[Dict]:
        """Return the latest face recognition results (safe copy)."""
        with self._frame_lock:
            return list(self._recognition_state)

    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture one frame for enrollment (without running detection).
        Returns BGR numpy array or None.
        """
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            return frame if ret else None

        # If camera isn't running, open briefly
        try:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Discard first few frames (camera warmup)
                for _ in range(5):
                    cap.read()
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
        except Exception as e:
            logger.error(f"Single frame capture error: {e}")
        return None

    def get_status(self) -> Dict:
        return {
            "status": self._status,
            "error": self._error,
            "camera_index": self.camera_index,
            "camera_source": self.camera_source,
            "camera_label": self.camera_label,
            "camera_type": "ip" if self.camera_source else "webcam",
            "frame_count": self._frame_count,
            "running": self._running,
        }
