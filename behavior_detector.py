"""
Real-time Human Behavior Detection from Webcam
Uses YOLOv8-Pose for person detection + pose estimation
Classifies: Standing, Walking, Reaching/Picking, Sitting, Falling
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

# ==================== CONFIG ====================
MODEL_NAME = "yolov8n-pose.pt"  # nano pose model (fastest)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Behavior smoothing window
SMOOTH_WINDOW = 5

# COCO keypoint indices (YOLOv8-pose)
KP = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# Skeleton connections for drawing
SKELETON = [
    (KP['left_shoulder'], KP['right_shoulder']),
    (KP['left_shoulder'], KP['left_elbow']),
    (KP['left_elbow'], KP['left_wrist']),
    (KP['right_shoulder'], KP['right_elbow']),
    (KP['right_elbow'], KP['right_wrist']),
    (KP['left_shoulder'], KP['left_hip']),
    (KP['right_shoulder'], KP['right_hip']),
    (KP['left_hip'], KP['right_hip']),
    (KP['left_hip'], KP['left_knee']),
    (KP['left_knee'], KP['left_ankle']),
    (KP['right_hip'], KP['right_knee']),
    (KP['right_knee'], KP['right_ankle']),
]

# Colors
COLOR_BOX = (0, 255, 255)      # Yellow
COLOR_KP = (0, 255, 0)         # Green
COLOR_SKEL = (255, 0, 255)     # Magenta
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)

# Behavior colors
BEHAVIOR_COLORS = {
    'Standing': (0, 255, 0),       # Green
    'Walking': (0, 255, 255),      # Cyan
    'Reaching/Picking': (0, 165, 255),  # Orange
    'Sitting': (255, 0, 255),      # Magenta
    'Falling': (0, 0, 255),        # Red
    'Unknown': (128, 128, 128),    # Gray
}


@dataclass
class PersonTrack:
    """Track a person across frames for temporal smoothing"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    keypoints: np.ndarray  # (17, 3) x, y, conf
    behavior: str
    behavior_history: deque
    center_history: deque  # For velocity calculation
    last_seen: float


class BehaviorClassifier:
    """Rule-based behavior classification from pose keypoints"""
    
    def __init__(self):
        self.behavior_buffer = {}
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points (p2 is vertex)"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def get_keypoint(self, kpts: np.ndarray, name: str) -> Optional[np.ndarray]:
        """Get keypoint by name, return None if low confidence"""
        idx = KP[name]
        if kpts[idx, 2] > 0.3:
            return kpts[idx]
        return None
    
    def compute_velocity(self, center_history: deque) -> float:
        """Compute movement velocity from center history"""
        if len(center_history) < 2:
            return 0.0
        positions = list(center_history)
        dists = []
        for i in range(1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
            dists.append(dist)
        return np.mean(dists) if dists else 0.0
    
    def classify(self, kpts: np.ndarray, center_history: deque) -> str:
        """
        Classify behavior based on pose keypoints and movement
        Returns behavior label
        """
        # Get key points
        nose = self.get_keypoint(kpts, 'nose')
        l_shoulder = self.get_keypoint(kpts, 'left_shoulder')
        r_shoulder = self.get_keypoint(kpts, 'right_shoulder')
        l_elbow = self.get_keypoint(kpts, 'left_elbow')
        r_elbow = self.get_keypoint(kpts, 'right_elbow')
        l_wrist = self.get_keypoint(kpts, 'left_wrist')
        r_wrist = self.get_keypoint(kpts, 'right_wrist')
        l_hip = self.get_keypoint(kpts, 'left_hip')
        r_hip = self.get_keypoint(kpts, 'right_hip')
        l_knee = self.get_keypoint(kpts, 'left_knee')
        r_knee = self.get_keypoint(kpts, 'right_knee')
        l_ankle = self.get_keypoint(kpts, 'left_ankle')
        r_ankle = self.get_keypoint(kpts, 'right_ankle')
        
        # Compute velocity
        velocity = self.compute_velocity(center_history)
        
        # --- FALLING DETECTION ---
        # Person horizontal (hip and shoulder at similar y)
        if l_hip is not None and r_hip is not None and l_shoulder is not None and r_shoulder is not None:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            if abs(hip_y - shoulder_y) < 50:  # Body horizontal
                return 'Falling'
        
        # --- SITTING DETECTION ---
        # Knees bent, hips lower than shoulders
        if l_hip is not None and r_hip is not None and l_knee is not None and r_knee is not None:
            hip_y = (l_hip[1] + r_hip[1]) / 2
            knee_y = (l_knee[1] + r_knee[1]) / 2
            if knee_y > hip_y + 30:  # Knees below hips (bent)
                # Check if ankles near knees (sitting)
                if l_ankle is not None and r_ankle is not None:
                    ankle_y = (l_ankle[1] + r_ankle[1]) / 2
                    if abs(ankle_y - knee_y) < 80:
                        return 'Sitting'
        
        # --- REACHING / PICKING DETECTION ---
        # Wrist below hip level, arm extended
        if l_wrist is not None and l_hip is not None and l_shoulder is not None and l_elbow is not None:
            # Left arm reaching down
            if l_wrist[1] > l_hip[1] + 20:
                # Check elbow angle (extended arm)
                angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
                if angle > 120:  # Arm relatively straight
                    return 'Reaching/Picking'
        
        if r_wrist is not None and r_hip is not None and r_shoulder is not None and r_elbow is not None:
            if r_wrist[1] > r_hip[1] + 20:
                angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                if angle > 120:
                    return 'Reaching/Picking'
        
        # --- WALKING DETECTION ---
        # Significant velocity + leg movement
        if velocity > 8:  # pixels per frame threshold
            # Check leg alternation (knees moving)
            if l_knee is not None and r_knee is not None and l_ankle is not None and r_ankle is not None:
                return 'Walking'
        
        # --- STANDING (DEFAULT) ---
        # Upright pose, low velocity
        if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
            hip_y = (l_hip[1] + r_hip[1]) / 2
            if shoulder_y < hip_y - 20:  # Shoulders above hips (upright)
                if velocity < 5:
                    return 'Standing'
        
        return 'Unknown'


class PersonTracker:
    """Simple IOU-based tracker for person ID persistence"""
    
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.tracks: List[PersonTrack] = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # frames
        self.classifier = BehaviorClassifier()
    
    def compute_iou(self, box1, box2):
        """Compute IOU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    def get_center(self, box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    
    def update(self, detections: List[Tuple], current_time: float) -> List[PersonTrack]:
        """
        detections: list of (bbox, keypoints)
        bbox: [x1, y1, x2, y2]
        keypoints: (17, 3) array
        """
        matched_tracks = set()
        matched_dets = set()
        
        # Match existing tracks to new detections
        for i, track in enumerate(self.tracks):
            best_iou = 0
            best_det = -1
            for j, (bbox, kpts) in enumerate(detections):
                if j in matched_dets:
                    continue
                iou = self.compute_iou(track.bbox, bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det = j
            
            if best_det >= 0:
                bbox, kpts = detections[best_det]
                track.bbox = bbox
                track.keypoints = kpts
                track.center_history.append(self.get_center(bbox))
                track.last_seen = current_time
                matched_tracks.add(i)
                matched_dets.add(best_det)
        
        # Create new tracks for unmatched detections
        for j, (bbox, kpts) in enumerate(detections):
            if j in matched_dets:
                continue
            center = self.get_center(bbox)
            new_track = PersonTrack(
                id=self.next_id,
                bbox=bbox,
                keypoints=kpts,
                behavior='Unknown',
                behavior_history=deque(maxlen=SMOOTH_WINDOW),
                center_history=deque([center], maxlen=10),
                last_seen=current_time
            )
            self.next_id += 1
            self.tracks.append(new_track)
        
        # Update behaviors and remove old tracks
        active_tracks = []
        for i, track in enumerate(self.tracks):
            if current_time - track.last_seen > self.max_age / 30:  # ~30 fps
                continue
            
            # Classify behavior
            behavior = self.classifier.classify(track.keypoints, track.center_history)
            track.behavior_history.append(behavior)
            
            # Smooth behavior (majority vote)
            if track.behavior_history:
                track.behavior = max(set(track.behavior_history), key=track.behavior_history.count)
            
            active_tracks.append(track)
        
        self.tracks = active_tracks
        return self.tracks


def draw_pose(frame: np.ndarray, kpts: np.ndarray, conf_threshold=0.3):
    """Draw pose skeleton on frame"""
    h, w = frame.shape[:2]
    
    # Draw keypoints
    for i in range(17):
        if kpts[i, 2] > conf_threshold:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            cv2.circle(frame, (x, y), 4, COLOR_KP, -1)
            cv2.circle(frame, (x, y), 6, (0, 0, 0), 1)
    
    # Draw skeleton
    for (i, j) in SKELETON:
        if kpts[i, 2] > conf_threshold and kpts[j, 2] > conf_threshold:
            x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
            x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
            cv2.line(frame, (x1, y1), (x2, y2), COLOR_SKEL, 2)


def draw_behavior_label(frame: np.ndarray, bbox: Tuple, behavior: str, track_id: int):
    """Draw behavior label above bounding box"""
    x1, y1, x2, y2 = bbox
    color = BEHAVIOR_COLORS.get(behavior, (128, 128, 128))
    
    # Label text
    label = f"ID:{track_id} {behavior}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Background rectangle
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 0), 1)
    
    # Text
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


def draw_info_panel(frame: np.ndarray, tracks: List[PersonTrack], fps: float):
    """Draw info panel on the right side"""
    h, w = frame.shape[:2]
    panel_w = 280
    panel_x = w - panel_w
    
    # Panel background
    cv2.rectangle(frame, (panel_x, 0), (w, h), (20, 20, 30), -1)
    cv2.line(frame, (panel_x, 0), (panel_x, h), (60, 60, 80), 2)
    
    # Title
    cv2.putText(frame, "BEHAVIOR ANALYSIS", (panel_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"People: {len(tracks)}", (panel_x + 10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Behavior counts
    behavior_counts = {}
    for t in tracks:
        behavior_counts[t.behavior] = behavior_counts.get(t.behavior, 0) + 1
    
    y = 120
    for behavior, count in behavior_counts.items():
        color = BEHAVIOR_COLORS.get(behavior, (128, 128, 128))
        cv2.circle(frame, (panel_x + 20, y), 6, color, -1)
        cv2.putText(frame, f"{behavior}: {count}", (panel_x + 35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y += 30
    
    # Legend
    y += 20
    cv2.putText(frame, "BEHAVIORS:", (panel_x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    
    for behavior, color in BEHAVIOR_COLORS.items():
        cv2.circle(frame, (panel_x + 20, y), 5, color, -1)
        cv2.putText(frame, behavior, (panel_x + 35, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        y += 25


def main():
    print("Loading YOLOv8-Pose model...")
    model = YOLO(MODEL_NAME)
    print(f"Model loaded: {MODEL_NAME}")
    
    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam {WEBCAM_INDEX}")
        return
    
    print("Webcam opened. Press 'q' to quit, 's' to save screenshot")
    
    tracker = PersonTracker()
    prev_time = time.time()
    fps = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.resize(frame, (FRAME_WIDTH + 280, FRAME_HEIGHT))
        
        # Run inference
        results = model(frame[:, :FRAME_WIDTH], conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        
        # Extract detections
        detections = []
        for r in results:
            if r.boxes is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                kpts = r.keypoints.xyn.cpu().numpy()  # normalized [0,1]
                kpts_conf = r.keypoints.conf.cpu().numpy()  # confidence
                
                # Convert normalized keypoints to pixel coordinates
                for i, (box, kpt, conf) in enumerate(zip(boxes, kpts, kpts_conf)):
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Denormalize keypoints
                    kpt_px = np.zeros((17, 3))
                    kpt_px[:, 0] = kpt[:, 0] * FRAME_WIDTH
                    kpt_px[:, 1] = kpt[:, 1] * FRAME_HEIGHT
                    kpt_px[:, 2] = conf
                    
                    detections.append(([x1, y1, x2, y2], kpt_px))
        
        # Update tracker
        current_time = time.time()
        tracks = tracker.update(detections, current_time)
        
        # Draw on frame
        display_frame = frame.copy()
        
        # Draw each person
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
            
            # Pose skeleton
            draw_pose(display_frame, track.keypoints)
            
            # Behavior label
            draw_behavior_label(display_frame, track.bbox, track.behavior, track.id)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - prev_time)
            prev_time = time.time()
        
        # Info panel
        draw_info_panel(display_frame, tracks, fps)
        
        # Show frame
        cv2.imshow("Human Behavior Detection - YOLOv8-Pose", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"behavior_capture_{timestamp}.jpg", display_frame)
            print(f"Screenshot saved: behavior_capture_{timestamp}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()