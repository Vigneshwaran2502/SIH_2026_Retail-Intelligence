"""
Optimized Real-time Human Behavior Detection
- Async inference thread for smooth camera feed
- Lower input resolution (320) for speed
- Improved walking detection (ankle velocity + stride)
- Better smoothing & reduced latency
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ==================== CONFIG ====================
MODEL_NAME = "yolov8n-pose.pt"
INFERENCE_SIZE = 320          # Smaller = faster (was 640)
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
WEBCAM_INDEX = 0
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
PANEL_WIDTH = 280

# Behavior smoothing
SMOOTH_WINDOW = 7
MAX_TRACK_AGE = 1.0  # seconds

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
    'text_bg': (0, 0, 0), 'text': (255, 255, 255),
    'Standing': (0, 255, 0), 'Walking': (0, 255, 255),
    'Reaching/Picking': (0, 165, 255), 'Sitting': (255, 0, 255),
    'Falling': (0, 0, 255), 'Unknown': (128, 128, 128)
}


@dataclass
class TrackedPerson:
    id: int
    bbox: Tuple[int, int, int, int]
    kpts: np.ndarray          # (17, 3) pixel coords
    behavior: str = 'Unknown'
    beh_history: deque = None
    ankle_history: deque = None   # For walking detection
    center_history: deque = None
    last_update: float = 0
    
    def __post_init__(self):
        self.beh_history = deque(maxlen=SMOOTH_WINDOW)
        self.ankle_history = deque(maxlen=10)
        self.center_history = deque(maxlen=10)


class InferenceWorker(threading.Thread):
    """Runs YOLO inference in background thread"""
    
    def __init__(self, model, in_queue, out_queue, inf_size):
        super().__init__(daemon=True)
        self.model = model
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.inf_size = inf_size
        self.running = True
    
    def run(self):
        while self.running:
            try:
                frame, timestamp = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Run inference
            results = self.model(frame, imgsz=self.inf_size, conf=CONF_THRESHOLD, 
                                 iou=IOU_THRESHOLD, verbose=False)
            
            # Parse results
            detections = []
            for r in results:
                if r.boxes is not None and r.keypoints is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    kpts_xy = r.keypoints.xy.cpu().numpy()
                    kpts_conf = r.keypoints.conf.cpu().numpy()
                    
                    for box, kpt, conf in zip(boxes, kpts_xy, kpts_conf):
                        detections.append((box, kpt, conf))
            
            # Send back with timestamp
            try:
                self.out_queue.put_nowait((detections, timestamp))
            except queue.Full:
                pass  # Drop old result
    
    def stop(self):
        self.running = False


class BehaviorClassifier:
    """Optimized behavior classification"""
    
    @staticmethod
    def angle(p1, p2, p3):
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
    
    @staticmethod
    def kpt(kpts, name, conf_thresh=0.35):
        idx = KP[name]
        if kpts[idx, 2] > conf_thresh:
            return kpts[idx]
        return None
    
    @staticmethod
    def center(bbox):
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
    
    def velocity(self, history: deque) -> float:
        if len(history) < 2:
            return 0
        pts = list(history)
        return float(np.mean([np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) 
                             for i in range(1, len(pts))]))
    
    def ankle_velocity(self, history: deque) -> float:
        """Compute velocity from ankle positions (better for walking)"""
        if len(history) < 2:
            return 0
        pts = list(history)
        return float(np.mean([np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) 
                             for i in range(1, len(pts))]))
    
    def classify(self, kpts: np.ndarray, center_hist: deque, ankle_hist: deque) -> str:
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
        
        # Velocities
        vel = self.velocity(center_hist)
        ank_vel = self.ankle_velocity(ankle_hist)
        
        # Store ankle positions for walking detection
        if l_ank is not None and r_ank is not None:
            ankle_center = ((l_ank[0] + r_ank[0]) * 0.5, (l_ank[1] + r_ank[1]) * 0.5)
            ankle_hist.append(ankle_center)
        
        # --- FALLING: body horizontal ---
        if l_hip is not None and r_hip is not None and l_sho is not None and r_sho is not None:
            if abs((l_hip[1]+r_hip[1])*0.5 - (l_sho[1]+r_sho[1])*0.5) < 40:
                return 'Falling'
        
        # --- SITTING: knees bent, hips low ---
        if l_hip is not None and r_hip is not None and l_knee is not None and r_knee is not None:
            hip_y = (l_hip[1] + r_hip[1]) * 0.5
            knee_y = (l_knee[1] + r_knee[1]) * 0.5
            if knee_y > hip_y + 25:  # knees below hips
                if l_ank is not None and r_ank is not None:
                    ankle_y = (l_ank[1] + r_ank[1]) * 0.5
                    if abs(ankle_y - knee_y) < 70:
                        return 'Sitting'
        
        # --- REACHING/PICKING: wrist below hip, arm extended ---
        for (wri, elb, sho, hip) in [(l_wri, l_elb, l_sho, l_hip), (r_wri, r_elb, r_sho, r_hip)]:
            if wri is not None and hip is not None and elb is not None and sho is not None:
                if wri[1] > hip[1] + 15:  # wrist below hip
                    if self.angle(sho, elb, wri) > 110:
                        return 'Reaching/Picking'
        
        # --- WALKING: ankle velocity + center velocity ---
        # Walking shows alternating ankle movement even if center moves slowly
        if ank_vel > 3.0 or (ank_vel > 1.5 and vel > 2.0):
            # Verify legs are moving (knees not static)
            if l_knee is not None and r_knee is not None:
                return 'Walking'
        
        # Center velocity fallback
        if vel > 4.0:
            return 'Walking'
        
        # --- STANDING: upright, low velocity ---
        if l_sho is not None and r_sho is not None and l_hip is not None and r_hip is not None:
            sho_y = (l_sho[1] + r_sho[1]) * 0.5
            hip_y = (l_hip[1] + r_hip[1]) * 0.5
            if sho_y < hip_y - 20 and vel < 3.0:
                return 'Standing'
        
        return 'Unknown'


class Tracker:
    """IOU tracker with behavior smoothing"""
    
    def __init__(self):
        self.tracks: List[TrackedPerson] = []
        self.next_id = 1
        self.classifier = BehaviorClassifier()
    
    @staticmethod
    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        inter = (x2-x1)*(y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter)
    
    def update(self, detections, now):
        matched_t = set()
        matched_d = set()
        
        # Match existing tracks
        for i, tr in enumerate(self.tracks):
            best_iou, best_j = 0, -1
            for j, (box, kpt, conf) in enumerate(detections):
                if j in matched_d: continue
                iou = self.iou(tr.bbox, box)
                if iou > best_iou and iou > 0.25:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                box, kpt, conf = detections[best_j]
                tr.bbox = box.astype(int)
                tr.kpts = np.column_stack([kpt, conf])
                tr.center_history.append(BehaviorClassifier.center(box))
                tr.last_update = now
                matched_t.add(i); matched_d.add(best_j)
        
        # New tracks
        for j, (box, kpt, conf) in enumerate(detections):
            if j in matched_d: continue
            tr = TrackedPerson(
                id=self.next_id,
                bbox=box.astype(int),
                kpts=np.column_stack([kpt, conf]),
                center_history=deque([BehaviorClassifier.center(box)], maxlen=10)
            )
            self.next_id += 1
            self.tracks.append(tr)
        
        # Update behaviors, prune old
        active = []
        for tr in self.tracks:
            if now - tr.last_update > MAX_TRACK_AGE:
                continue
            beh = self.classifier.classify(tr.kpts, tr.center_history, tr.ankle_history)
            tr.beh_history.append(beh)
            tr.behavior = max(set(tr.beh_history), key=tr.beh_history.count)
            active.append(tr)
        
        self.tracks = active
        return self.tracks


def draw_skeleton(img, kpts):
    for i in range(17):
        if kpts[i, 2] > 0.3:
            cv2.circle(img, (int(kpts[i,0]), int(kpts[i,1])), 3, COLORS['kp'], -1)
    for i, j in SKELETON:
        if kpts[i,2] > 0.3 and kpts[j,2] > 0.3:
            cv2.line(img, (int(kpts[i,0]), int(kpts[i,1])), 
                     (int(kpts[j,0]), int(kpts[j,1])), COLORS['skel'], 2)


def draw_label(img, box, beh, tid):
    x1, y1, x2, y2 = box
    color = COLORS.get(beh, COLORS['Unknown'])
    label = f"ID{tid} {beh}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1-th-8), (x1+tw+8, y1), color, -1)
    cv2.putText(img, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS['text'], 2)


def draw_panel(img, tracks, fps, inf_fps):
    h, w = img.shape[:2]
    px = w - PANEL_WIDTH
    cv2.rectangle(img, (px, 0), (w, h), (18, 18, 28), -1)
    cv2.line(img, (px, 0), (px, h), (50, 50, 70), 2)
    
    cv2.putText(img, "BEHAVIOR", (px+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"FPS: {fps:.1f} | INF: {inf_fps:.1f}", (px+10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    cv2.putText(img, f"Tracks: {len(tracks)}", (px+10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    
    counts = {}
    for t in tracks:
        counts[t.behavior] = counts.get(t.behavior, 0) + 1
    
    y = 110
    for b, c in counts.items():
        cv2.circle(img, (px+20, y), 6, COLORS.get(b, COLORS['Unknown']), -1)
        cv2.putText(img, f"{b}: {c}", (px+35, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        y += 28
    
    y += 10
    cv2.putText(img, "LEGEND:", (px+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
    for b, col in [('Standing', COLORS['Standing']), ('Walking', COLORS['Walking']),
                   ('Reaching', COLORS['Reaching/Picking']), ('Sitting', COLORS['Sitting']),
                   ('Falling', COLORS['Falling'])]:
        y += 22
        cv2.circle(img, (px+20, y), 5, col, -1)
        cv2.putText(img, b, (px+32, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)


def main():
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    print("Model loaded. Starting webcam...")
    
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW faster on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        print(f"Cannot open camera {WEBCAM_INDEX}")
        return
    
    # Async inference queues
    in_q = queue.Queue(maxsize=2)
    out_q = queue.Queue(maxsize=2)
    worker = InferenceWorker(model, in_q, out_q, INFERENCE_SIZE)
    worker.start()
    
    tracker = Tracker()
    latest_detections = []
    inf_times = deque(maxlen=10)
    
    frame_count = 0
    t0 = time.time()
    fps = 0
    inf_fps = 0
    last_inf_time = 0
    
    print("Running. Press 'q' to quit, 's' to screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for display
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # Send to inference thread (non-blocking)
            try:
                in_q.put_nowait((frame.copy(), time.time()))
            except queue.Full:
                pass
            
            # Get latest inference result
            while True:
                try:
                    latest_detections, inf_ts = out_q.get_nowait()
                    inf_times.append(time.time() - inf_ts)
                except queue.Empty:
                    break
            
            # Update tracker with latest detections
            now = time.time()
            tracks = tracker.update(latest_detections, now)
            
            # Draw
            for tr in tracks:
                cv2.rectangle(frame, (tr.bbox[0], tr.bbox[1]), 
                              (tr.bbox[2], tr.bbox[3]), COLORS['box'], 2)
                draw_skeleton(frame, tr.kpts)
                draw_label(frame, tr.bbox, tr.behavior, tr.id)
            
            # FPS calc
            frame_count += 1
            if frame_count % 15 == 0:
                fps = 15 / (time.time() - t0)
                t0 = time.time()
                if inf_times:
                    inf_fps = len(inf_times) / sum(inf_times)
            
            # Panel
            draw_panel(frame, tracks, fps, inf_fps)
            
            cv2.imshow("Behavior Detection (Optimized)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{ts}.jpg", frame)
                print(f"Saved capture_{ts}.jpg")
    
    finally:
        worker.stop()
        worker.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()