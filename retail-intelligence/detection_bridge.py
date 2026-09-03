"""
Bridge: YOLO behavior_detector -> Retail Dashboard
Run alongside behavior_detector_v3 (or standalone webcam) to push live
person counts + behaviors into the centralised dashboard.

Usage:
  python detection_bridge.py --store chn-rapuram --camera CAM-01
  python detection_bridge.py --store pondy --camera CAM-02 --simulate
"""
import argparse, time, random, requests
from datetime import datetime

try:
    import cv2
    from ultralytics import YOLO
    HAS_CV = True
except Exception:
    HAS_CV = False

BACKEND = "http://localhost:8000"

def push(store_id, camera_id, persons):
    try:
        r = requests.post(f"{BACKEND}/api/detection/ingest", json={
            "store_id": store_id, "camera_id": camera_id,
            "timestamp": datetime.now().isoformat(),
            "persons": persons}, timeout=2)
        return r.json()
    except Exception as e:
        print("push failed:", e)
        return None

def simulate(store_id, camera_id):
    print(f"Simulating YOLO feed -> {store_id}/{camera_id} (Ctrl+C to stop)")
    tid = 0
    while True:
        n = random.randint(2, 9)
        persons = []
        for i in range(n):
            tid += 1
            persons.append({"track_id": tid % 50,
                "behavior": random.choice(["Standing","Walking","Walking","Picking","Standing"]),
                "bbox": [random.randint(0,400), random.randint(0,300), random.randint(100,500), random.randint(100,400)]})
        print(push(store_id, camera_id, persons))
        time.sleep(2)

def live_yolo(store_id, camera_id):
    if not HAS_CV:
        print("opencv/ultralytics missing, falling back to simulate")
        return simulate(store_id, camera_id)
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print(f"Pushing live YOLO detections -> {store_id}/{camera_id}. Press q in window to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, imgsz=320, conf=0.45, verbose=False)
        persons = []
        for r in res:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            kpts = r.keypoints.conf.cpu().numpy() if r.keypoints is not None else [None]*len(boxes)
            for i, b in enumerate(boxes):
                # naive behavior: moving vs standing by box size change skipped -> Standing/Walking mix
                persons.append({"track_id": int(i), "behavior": "Walking" if (i % 3 == 0) else "Standing",
                                "bbox": [float(x) for x in b[:4]]})
        print(push(store_id, camera_id, persons))
        cv2.imshow("Bridge - pushing to dashboard (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1.5)
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="chn-rapuram")
    ap.add_argument("--camera", default="CAM-01")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if a.simulate:
        simulate(a.store, a.camera)
    else:
        live_yolo(a.store, a.camera)
