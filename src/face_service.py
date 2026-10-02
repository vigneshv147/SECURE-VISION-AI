"""
face_service.py
----------------
Face detection and recognition service.

ARCHITECTURE (based on data.yaml inspection):
- YOLO model class: 'know' (nc=1) — detects enrolled/known faces
- A secondary lightweight face detector (OpenCV DNN or face_recognition)
  is used to catch faces that YOLO marks as unknown
- Recognition pipeline:
    Camera frame → YOLO inference → bounding boxes
    For each box: crop → face_recognition encoding → cosine similarity → user lookup

IMPORTANT:
- YOLO .pt must be placed at models/yolo/best.pt
- If model file missing, service degrades gracefully (camera still works, NIDS still works)
- Face embeddings: 128-dimensional face_recognition vectors
- Recognition threshold: configurable via FACE_MATCH_THRESHOLD (.env)
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed — YOLO detection unavailable")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed — camera unavailable")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not installed — using YOLO detections only")


class FaceService:
    """
    Singleton-style service for YOLO detection + face recognition.
    Call FaceService.get_instance() to get the shared instance.
    """

    _instance: Optional["FaceService"] = None

    def __init__(self):
        self.yolo_model = None
        self.yolo_class_names: List[str] = []
        self.yolo_confidence: float = float(os.environ.get("YOLO_CONFIDENCE", "0.50"))
        self.yolo_iou: float = float(os.environ.get("YOLO_IOU", "0.45"))
        self.face_match_threshold: float = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.50"))
        self.model_path: str = os.environ.get("YOLO_MODEL_PATH", "models/yolo/best.pt")
        self.data_yaml_path: str = os.environ.get("YOLO_DATA_PATH", "AICS.v1i.yolov11/data.yaml")
        self._status: Dict[str, str] = {}
        self._load_yolo()

    @classmethod
    def get_instance(cls) -> "FaceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─── YOLO Loading ────────────────────────────────────────────────────────

    def _load_yolo(self):
        """Attempt to load YOLO model. Sets status flags, never crashes."""
        if not ULTRALYTICS_AVAILABLE:
            self._status["yolo"] = "OFFLINE — ultralytics not installed"
            return

        if not os.path.exists(self.model_path):
            self._status["yolo"] = f"OFFLINE — model not found at {self.model_path}"
            logger.warning(self._status["yolo"])
            return

        try:
            self.yolo_model = YOLO(self.model_path)
            # Read class names from model (authoritative source)
            self.yolo_class_names = list(self.yolo_model.names.values())
            self._status["yolo"] = "ONLINE"
            logger.info(
                f"YOLO loaded: {self.model_path} | classes={self.yolo_class_names}"
            )
        except Exception as e:
            self._status["yolo"] = f"OFFLINE — {e}"
            logger.error(f"YOLO load error: {e}")

    def reload_yolo(self):
        """Reload YOLO model (e.g., after user places a new best.pt)."""
        self.yolo_model = None
        self._load_yolo()

    # ─── Inference ───────────────────────────────────────────────────────────

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Run YOLO inference on a single BGR frame.

        Returns list of detections:
        [
          {
            "bbox": (x1, y1, x2, y2),
            "class_id": 0,
            "class_name": "know",
            "confidence": 0.92,
            "is_known_class": True
          }, ...
        ]
        """
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(
                frame,
                conf=self.yolo_confidence,
                iou=self.yolo_iou,
                verbose=False
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.yolo_class_names[cls_id] if cls_id < len(self.yolo_class_names) else "unknown"
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": float(box.conf[0]),
                        "is_known_class": cls_name.lower() == "know",
                    })
            return detections
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []

    # ─── Face Recognition ────────────────────────────────────────────────────

    def generate_embedding(self, face_crop: np.ndarray) -> Optional[List[float]]:
        """
        Generate a 128-dim face_recognition encoding from a cropped face image.
        face_crop is a BGR numpy array (from OpenCV).
        Returns None if face_recognition not available or no face found in crop.
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return None

        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                return encodings[0].tolist()
            return None
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None

    def compare_embedding(
        self, query_embedding: List[float], known_embeddings: Dict[str, Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Compare query embedding against all known embeddings using cosine distance.
        
        Returns (user_id, similarity_score) of best match,
        or (None, 0.0) if no match above threshold.
        """
        if not query_embedding or not known_embeddings:
            return None, 0.0

        q = np.array(query_embedding)
        best_uid = None
        best_sim = -1.0

        for uid, data in known_embeddings.items():
            k = np.array(data["embedding"])
            # Cosine similarity
            norm_q = np.linalg.norm(q)
            norm_k = np.linalg.norm(k)
            if norm_q == 0 or norm_k == 0:
                continue
            sim = float(np.dot(q, k) / (norm_q * norm_k))
            if sim > best_sim:
                best_sim = sim
                best_uid = uid

        if best_sim >= self.face_match_threshold:
            return best_uid, best_sim
        return None, best_sim

    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and return the face region from a frame, with padding."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2]

    def draw_detections(
        self, frame: np.ndarray, detections: List[Dict],
        recognition_results: List[Dict]
    ) -> np.ndarray:
        """
        Draw bounding boxes + labels on frame.
        recognition_results is a list of dicts with keys: user_id, full_name, confidence, status
        """
        if not CV2_AVAILABLE:
            return frame

        annotated = frame.copy()
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            rec = recognition_results[i] if i < len(recognition_results) else {}
            status = rec.get("status", "UNKNOWN")

            if status == "AUTHENTICATED":
                color = (0, 220, 80)      # Green
                label = f"{rec.get('full_name', 'Unknown')} ({rec.get('confidence', 0)*100:.0f}%)"
            elif status == "UNKNOWN":
                color = (0, 60, 220)      # Red
                label = "UNKNOWN PERSON"
            else:
                color = (200, 150, 0)     # Orange — detected but not confirmed
                label = det.get("class_name", "face")

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        return annotated

    # ─── Status ──────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "yolo": self._status.get("yolo", "UNKNOWN"),
            "face_recognition": "ONLINE" if FACE_RECOGNITION_AVAILABLE else "OFFLINE — face_recognition not installed",
            "model_path": self.model_path,
            "class_names": self.yolo_class_names,
            "confidence_threshold": self.yolo_confidence,
            "match_threshold": self.face_match_threshold,
        }
