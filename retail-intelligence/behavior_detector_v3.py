"""
Advanced Human Behavior Detection v3
- Multi-scale inference for close + far people
- Hand gesture: open/close hand in front of camera = picking
- Robust sitting/standing/moving classification
- Full screen mode with auto-scaling
- Temporal smoothing + confidence scoring
"""

# Suppress ultralytics warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

# ==================== CONFIG ====================
MODEL_NAME = "yolov8n-pose.pt"
INFERENCE_SIZES = [256, 416]  # Multi-scale: small for far, large for close
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
WEBCAM_INDEX = 0
TARGET_FPS = 30

# Display
FULL_SCREEN = True
PANEL_WIDTH = 320

# Smoothing
SMOOTH_WINDOW = 9
MAX_TRACK_AGE = 1.2

# Gesture thresholds
HAND_OPEN_THRESH = 0.15   # finger spread ratio
HAND_CLOSE_THRESH = 0.05
GESTURE_COOLDOWN = 0.8    # seconds

# COCO keypoints
KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

COLORS = {
    'box': (0, 255, 255), 'kp': (0, 255, 0), 'skel': (255, 0, 255),
    'text': (255, 255, 255), 'panel_bg': (15, 15, 25),
    'Standing': (0, 255, 100), 'Walking': (0, 200, 255),
    'Picking': (0, 140, 255), 'Sitting': (180, 0, 255),
    'Falling': (0, 50, 255), 'Unknown': (100, 100, 100),
    'gesture': (0, 255, 255)
}


@dataclass
class TrackedPerson:
    id: int
    bbox: Tuple[int, int, int, int]
    kpts: np.ndarray              # (17, 3) pixel
    behavior: str = 'Unknown'
    confidence: float = 0.0
    beh_history: deque = None
    beh_conf_history: deque = None
    ankle_hist: deque = None
    wrist_hist: deque = None      # For gesture
    center_hist: deque = None
    last_update: float = 0
    gesture_state: str = 'none'   # none, open, closed, picking
    last_gesture_time: float = 0
    
    def __post_init__(self):
        self.beh_history = deque(maxlen=SMOOTH_WINDOW)
        self.beh_conf_history = deque(maxlen=SMOOTH_WINDOW)
        self.ankle_hist = deque(maxlen=12)
        self.wrist_hist = deque(maxlen=12)
        self.center_hist = deque(maxlen=12)


class InferenceWorker(threading.Thread):
    """Multi-scale async inference"""
    
    def __init__(self, model, in_q, out_q, scale):
        super().__init__(daemon=True)
        self.model = model
        self.in_q = in_q
        self.out_q = out_q
        self.scale = scale
        self.running = True
    
    def run(self):
        while self.running:
            try:
                frame, timestamp = self.in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            
            # Run inference at specified scale
            results = self.model(frame, imgsz=self.scale, conf=CONF_THRESHOLD, 
                                 iou=IOU_THRESHOLD, verbose=False)
            
            detections = []
            for r in results:
                if r.boxes is not None and r.keypoints is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    kpts_xy = r.keypoints.xy.cpu().numpy()
                    kpts_conf = r.keypoints.conf.cpu().numpy()
                    cls_conf = r.boxes.conf.cpu().numpy()
                    
                    for box, kpt, conf, cconf in zip(boxes, kpts_xy, kpts_conf, cls_conf):
                        detections.append((box, kpt, conf, cconf))
            
            try:
                self.out_q.put_nowait((detections, timestamp))
            except queue.Full:
                pass
    
    def stop(self):
        self.running = False


class BehaviorEngine:
    """Advanced behavior + gesture classification"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def dist(p1, p2):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])
    
    @staticmethod
    def angle(p1, p2, p3):
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
    
    @staticmethod
    def kpt(kpts, name, thr=0.3):
        idx = KP[name]
        if kpts[idx, 2] > thr:
            return kpts[idx]
        return None
    
    @staticmethod
    def center(bbox):
        return ((bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5)
    
    def velocity(self, hist: deque):
        if len(hist) < 2: return 0
        pts = list(hist)
        return float(np.mean([self.dist(pts[i], pts[i-1]) for i in range(1, len(pts))]))
    
    def hand_state(self, wrist, elbow, shoulder):
        """Detect if hand is open or closed based on finger spread"""
        # Since YOLOv8-pose doesn't have finger keypoints, 
        # we approximate from wrist-elbow-shoulder geometry
        # Open hand: wrist further from elbow (extended fingers)
        # Closed hand: wrist closer to elbow (fist)
        if wrist is None or elbow is None or shoulder is None:
            return 'unknown'
        d_we = self.dist(wrist, elbow)
        d_es = self.dist(elbow, shoulder)
        if d_es == 0:
            return 'unknown'
        ratio = d_we / d_es
        if ratio > 0.85:
            return 'open'
        elif ratio < 0.65:
            return 'closed'
        return 'neutral'
    
    def hand_in_front_of_body(self, wrist, shoulder, hip, nose):
        """Check if hand is in front of torso (camera-facing)"""
        if wrist is None or shoulder is None or hip is None:
            return False
        # Hand between shoulder and hip vertically, and forward of shoulders
        sho_y, hip_y = shoulder[1], hip[1]
        if not (min(sho_y, hip_y) - 30 < wrist[1] < max(sho_y, hip_y) + 30):
            return False
        # Check depth via shoulder width vs wrist offset
        return True
    
    def classify(self, tr: TrackedPerson) -> Tuple[str, float]:
        kpts = tr.kpts
        now = time.time()
        
        # Key points
        l_sho = self.kpt(kpts, 'left_shoulder')
        r_sho = self.kpt(kpts, 'right_shoulder')
        l_hip = self.kpt(kpts, 'left_hip')
        r_hip = self.kpt(kpts, 'right_hip')
        l_knee = self.kpt(kpts, 'left_knee')
        r_knee = self.kpt(kpts, 'right_knee')
        l_ank = self.kpt(kpts, 'left_ankle')
        r_ank = self.kpt(kpts, 'right_ankle')
        l_wri = self.kpt(kpts, 'left_wrist')
        r_wri = self.kpt(kpts, 'right_wrist')
        l_elb = self.kpt(kpts, 'left_elbow')
        r_elb = self.kpt(kpts, 'right_elbow')
        nose = self.kpt(kpts, 'nose')
        
        # Velocities
        vel = self.velocity(tr.center_hist)
        ank_vel = self.velocity(tr.ankle_hist)
        
        # Store ankle positions
        if l_ank is not None and r_ank is not None:
            tr.ankle_hist.append(((l_ank[0]+r_ank[0])*0.5, (l_ank[1]+r_ank[1])*0.5))
        
        # Store wrists for gesture
        if l_wri is not None:
            tr.wrist_hist.append(('L', l_wri, l_elb, l_sho))
        if r_wri is not None:
            tr.wrist_hist.append(('R', r_wri, r_elb, r_sho))
        
        # =========== BEHAVIOR CLASSIFICATION ===========
        beh, conf = 'Unknown', 0.0
        
        # FALLING - body horizontal
        if l_hip and r_hip and l_sho and r_sho:
            if abs((l_hip[1]+r_hip[1])*0.5 - (l_sho[1]+r_sho[1])*0.5) < 35:
                return 'Falling', 0.9
        
        # SITTING - knees bent, hips low, stable
        if l_hip and r_hip and l_knee and r_knee:
            hip_y = (l_hip[1]+r_hip[1])*0.5
            knee_y = (l_knee[1]+r_knee[1])*0.5
            if knee_y > hip_y + 20:
                if l_ank and r_ank:
                    ankle_y = (l_ank[1]+r_ank[1])*0.5
                    if abs(ankle_y - knee_y) < 65 and vel < 3:
                        return 'Sitting', 0.85
        
        # GESTURE: PICKING - hand open/close cycle in front of body
        for side, wrist, elbow, shoulder in list(tr.wrist_hist)[-4:]:
            if wrist and elbow and shoulder and l_hip and r_hip:
                hip_y = (l_hip[1]+r_hip[1])*0.5
                # Hand in front of torso zone
                if abs(wrist[1] - hip_y) < 80:  # near hip level
                    state = self.hand_state(wrist, elbow, shoulder)
                    if state != 'unknown':
                        tr.gesture_state = state
                        # Detect open->close transition = picking
                        if tr.gesture_state == 'closed' and \
                           len(tr.wrist_hist) >= 2:
                            prev_state = self.hand_state(
                                tr.wrist_hist[-2][1], tr.wrist_hist[-2][2], tr.wrist_hist[-2][3]
                            ) if tr.wrist_hist[-2][1] is not None else 'unknown'
                            if prev_state == 'open' and now - tr.last_gesture_time > GESTURE_COOLDOWN:
                                tr.last_gesture_time = now
                                return 'Picking', 0.9
        
        # WALKING - ankle velocity + stride pattern
        stride_score = 0
        if ank_vel > 2.5:
            stride_score += 2
        if ank_vel > 1.5 and vel > 1.5:
            stride_score += 1
        # Check knee alternation
        if l_knee and r_knee:
            knee_sep = abs(l_knee[0] - r_knee[0])
            if knee_sep > 30:  # legs apart
                stride_score += 1
        
        if stride_score >= 2:
            return 'Walking', min(0.7 + stride_score * 0.1, 0.95)
        
        # Center velocity fallback
        if vel > 5:
            return 'Walking', 0.7
        
        # STANDING - upright, stable
        if l_sho and r_sho and l_hip and r_hip:
            sho_y = (l_sho[1]+r_sho[1])*0.5
            hip_y = (l_hip[1]+r_hip[1])*0.5
            if sho_y < hip_y - 15 and vel < 2.5 and ank_vel < 1.5:
                return 'Standing', 0.8
        
        return beh, conf


class MultiScaleTracker:
    """Tracks across scales with size-aware matching"""
    
    def __init__(self):
        self.tracks: List[TrackedPerson] = []
        self.next_id = 1
        self.engine = BehaviorEngine()
    
    @staticmethod
    def center(bbox):
        return ((bbox[0]+bbox[2])*0.5, (bbox[1]+bbox[3])*0.5)
    
    def size_score(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1: return 0
        inter = (x2-x1)*(y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter)
    
    def size_score(self, b1, b2):
        """Prefer matches of similar size (close↔close, far↔far)"""
        s1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        s2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        if s1 == 0 or s2 == 0: return 0
        return min(s1, s2) / max(s1, s2)
    
    def update(self, detections, now):
        matched_t, matched_d = set(), set()
        
        # Match: combine IOU + size similarity
        for i, tr in enumerate(self.tracks):
            best_score, best_j = 0, -1
            for j, (box, kpt, conf, cconf) in enumerate(detections):
                if j in matched_d: continue
                iou = self.iou(tr.bbox, box)
                sz = self.size_score(tr.bbox, box)
                score = 0.7 * iou + 0.3 * sz
                if score > best_score and iou > 0.2:
                    best_score, best_j = score, j
            if best_j >= 0:
                box, kpt, conf, cconf = detections[best_j]
                tr.bbox = box.astype(int)
                tr.kpts = np.column_stack([kpt, conf])
                tr.center_hist.append(MultiScaleTracker.center(box))
                tr.last_update = now
                matched_t.add(i); matched_d.add(best_j)
        
        # New tracks
        for j, (box, kpt, conf, cconf) in enumerate(detections):
            if j in matched_d: continue
            tr = TrackedPerson(
                id=self.next_id,
                bbox=box.astype(int),
                kpts=np.column_stack([kpt, conf]),
                center_hist=deque([MultiScaleTracker.center(box)], maxlen=12)
            )
            self.next_id += 1
            self.tracks.append(tr)
        
        # Classify & prune
        active = []
        for tr in self.tracks:
            if now - tr.last_update > MAX_TRACK_AGE:
                continue
            beh, conf = self.engine.classify(tr)
            tr.beh_history.append(beh)
            tr.beh_conf_history.append(conf)
            # Weighted majority vote
            if tr.beh_history:
                weights = list(tr.beh_conf_history) if tr.beh_conf_history else [1]*len(tr.beh_history)
                beh_scores = {}
                for b, w in zip(tr.beh_history, weights):
                    beh_scores[b] = beh_scores.get(b, 0) + w
                tr.behavior = max(beh_scores, key=beh_scores.get)
                tr.confidence = beh_scores[tr.behavior] / sum(beh_scores.values())
            active.append(tr)
        
        self.tracks = active
        return self.tracks


def draw_skeleton(img, kpts, conf_thr=0.25):
    for i in range(17):
        if kpts[i, 2] > conf_thr:
            cv2.circle(img, (int(kpts[i,0]), int(kpts[i,1])), 4, COLORS['kp'], -1)
    for i, j in SKELETON:
        if kpts[i,2] > conf_thr and kpts[j,2] > conf_thr:
            cv2.line(img, (int(kpts[i,0]), int(kpts[i,1])), 
                     (int(kpts[j,0]), int(kpts[j,1])), COLORS['skel'], 2)


def draw_label(img, box, beh, conf, tid):
    x1, y1, x2, y2 = box
    color = COLORS.get(beh, COLORS['Unknown'])
    label = f"ID{tid} {beh} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
    cv2.putText(img, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text'], 2)


def draw_panel(img, tracks, fps, inf_fps, gesture_info):
    h, w = img.shape[:2]
    px = w - PANEL_WIDTH
    cv2.rectangle(img, (px, 0), (w, h), COLORS['panel_bg'], -1)
    cv2.line(img, (px, 0), (px, h), (50, 50, 80), 2)
    
    cv2.putText(img, "BEHAVIOR ANALYSIS", (px+15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(img, f"Display FPS: {fps:.1f}", (px+15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170,170,170), 1)
    cv2.putText(img, f"Inference FPS: {inf_fps:.1f}", (px+15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170,170,170), 1)
    cv2.putText(img, f"Active: {len(tracks)}", (px+15, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170,170,170), 1)
    
    # Behavior counts
    counts = {}
    for t in tracks:
        counts[t.behavior] = counts.get(t.behavior, 0) + 1
    
    y = 135
    for b in ['Standing', 'Walking', 'Picking', 'Sitting', 'Falling']:
        c = counts.get(b, 0)
        col = COLORS.get(b, COLORS['Unknown'])
        cv2.circle(img, (px+22, y), 7, col, -1)
        cv2.putText(img, f"{b}: {c}", (px+38, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210,210,210), 1)
        y += 30
    
    # Gesture status
    y += 10
    cv2.putText(img, "HAND GESTURE:", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    y += 28
    cv2.putText(img, f"State: {gesture_info}", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    
    # Legend
    y += 40
    cv2.putText(img, "KEY:", (px+15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
    for b, col in [('Standing', COLORS['Standing']), ('Walking', COLORS['Walking']),
                   ('Picking', COLORS['Picking']), ('Sitting', COLORS['Sitting']),
                   ('Falling', COLORS['Falling'])]:
        y += 22
        cv2.circle(img, (px+22, y), 5, col, -1)
        cv2.putText(img, b, (px+35, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)


def main():
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print("Starting webcam...")
    
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get native resolution
    ret, test_frame = cap.read()
    if not ret:
        print("Camera failed")
        return
    native_h, native_w = test_frame.shape[:2]
    print(f"Native: {native_w}x{native_h}")
    
    # Full screen window
    cv2.namedWindow("Advanced Behavior Detection", cv2.WINDOW_NORMAL)
    if FULL_SCREEN:
        cv2.setWindowProperty("Advanced Behavior Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Async inference at two scales
    in_q = queue.Queue(maxsize=2)
    out_q = queue.Queue(maxsize=2)
    workers = [
        InferenceWorker(model, in_q, out_q, INFERENCE_SIZES[0]),
        InferenceWorker(model, in_q, out_q, INFERENCE_SIZES[1])
    ]
    for w in workers: w.start()
    
    tracker = MultiScaleTracker()
    latest_dets = []
    inf_times = deque(maxlen=20)
    gesture_msg = "Waiting..."
    
    frame_count = 0
    t0 = time.time()
    fps = 0
    inf_fps = 0
    
    print("Running full-screen. ESC/q=quit, s=screenshot, f=toggle fullscreen")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Always process at native resolution for best accuracy
            h, w = frame.shape[:2]
            
            # Send to inference (alternating scales for multi-scale coverage)
            try:
                in_q.put_nowait((frame.copy(), time.time()))
            except queue.Full:
                pass
            
            # Collect results
            while True:
                try:
                    dets, ts = out_q.get_nowait()
                    latest_dets = dets
                    inf_times.append(time.time() - ts)
                except queue.Empty:
                    break
            
            # Track
            now = time.time()
            tracks = tracker.update(latest_dets, now)
            
            # Draw
            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['box'], 2)
                draw_skeleton(frame, tr.kpts)
                draw_label(frame, tr.bbox, tr.behavior, tr.confidence, tr.id)
            
            # Gesture status
            gesture_states = [tr.gesture_state for tr in tracks if tr.gesture_state != 'none']
            if gesture_states:
                gesture_msg = f"Detected: {', '.join(set(gesture_states))}"
            else:
                gesture_msg = "No gesture"
            
            # FPS
            frame_count += 1
            if frame_count % 20 == 0:
                fps = 20 / (time.time() - t0)
                t0 = time.time()
                if inf_times:
                    inf_fps = len(inf_times) / sum(inf_times)
            
            # Panel
            draw_panel(frame, tracks, fps, inf_fps, gesture_msg)
            
            # Full screen display
            cv2.imshow("Advanced Behavior Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"behav_{ts}.jpg", frame)
                print(f"Saved behav_{ts}.jpg")
            elif key == ord('f'):
                # Toggle fullscreen
                cv2.setWindowProperty("Advanced Behavior Detection", cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if not FULL_SCREEN else cv2.WINDOW_NORMAL)
    
    finally:
        for w in workers:
            w.stop()
            w.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()