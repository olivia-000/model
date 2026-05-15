from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import requests
import threading
import time
from datetime import datetime
from collections import deque
from ultralytics import YOLO
 
app = Flask(__name__)
 
# ===== 設定區 =====
ESP32_IP = "172.20.10.2"
CAPTURE_URL = f"http://{ESP32_IP}/capture"
LED_URL = f"http://{ESP32_IP}/led"  # LED 控制網址
MODEL_PATH = r"C:\Users\jarvi\Desktop\人工智慧專題\runs\person_yolo_train\weights\best.pt"
LOITER_SECONDS = 5.0
CONF_THRESHOLD = 0.2
ROI_POINTS = [(0, 0), (1280, 0), (1280, 960), (0, 960)]
TRAIL_MAXLEN = 60
 
# ===== 全域狀態 =====
latest_frame_bbox = None
latest_frame_traj = None
latest_frame_heat = None
frame_lock = threading.Lock()
alert_log = deque(maxlen=100)
alert_log_lock = threading.Lock()
hourly_counts = [0] * 24
trajectories = {}
behaviors = {}
heatmap_acc = None
heatmap_lock = threading.Lock()
system_status = {
    "camera": "connecting",
    "model": "loading",
    "fps": 0,
    "active_alerts": 0,
    "total_today": 0,
    "persons_detected": 0,
    "esp32_ip": ESP32_IP,
    "loiter_threshold": LOITER_SECONDS,
    "conf_threshold": CONF_THRESHOLD,
}
entered_at = {}
last_seen_at = {}
alerted_ids = set()
fps_counter = {"count": 0, "last_time": time.time()}
 
TRAIL_COLORS = [
    (56, 189, 248), (167, 139, 250), (251, 146, 60),
    (52, 211, 153), (244, 114, 182), (250, 204, 21),
]
 
# ===== LED 控制函數 =====
def set_led(state: str):
    # 發送 HTTP 請求給 ESP32 控制 LED
    try:
        requests.get(f"{LED_URL}?state={state}", timeout=2)
    except:
        pass  # 失敗不影響主程式
 
def point_in_polygon(point, polygon):
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, point, False) >= 0
 
def classify_behavior(trail):
    pts = list(trail)
    if len(pts) < 10:
        return "tracking"
    total_dist = sum(
        ((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)**0.5
        for i in range(1, len(pts))
    )
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    straight = (dx**2 + dy**2)**0.5
    if straight < 5:
        return "stationary"
    ratio = total_dist / straight
    if ratio > 3.5:
        return "wandering"
    elif ratio > 1.8:
        return "circling"
    else:
        return "traversing"
 
def fetch_frame():
    try:
        r = requests.get(CAPTURE_URL, timeout=5)
        if r.status_code == 200 and len(r.content) > 0:
            img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img
    except:
        pass
    return None
 
def update_heatmap(frame, cx, cy):
    global heatmap_acc
    with heatmap_lock:
        if heatmap_acc is None:
            heatmap_acc = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        heatmap_acc *= 0.995
        cv2.circle(heatmap_acc, (cx, cy), 30, 1.0, -1)
 
def render_heatmap(frame):
    with heatmap_lock:
        if heatmap_acc is None or heatmap_acc.max() == 0:
            return frame.copy()
        norm = cv2.normalize(heatmap_acc, None, 0, 255, cv2.NORM_MINMAX)
        norm = np.uint8(norm)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        result = cv2.addWeighted(frame, 0.45, colored, 0.55, 0)
        return result
 
def detection_loop():
    global latest_frame_bbox, latest_frame_traj, latest_frame_heat, system_status
    model = YOLO(MODEL_PATH)
    system_status["model"] = "loaded"
 
    while True:
        frame = fetch_frame()
        if frame is None:
            system_status["camera"] = "disconnected"
            time.sleep(0.2)
            continue
 
        system_status["camera"] = "connected"
        now = time.time()
 
        results = model.track(frame, conf=CONF_THRESHOLD, classes=[0],
                              tracker="bytetrack.yaml", persist=True, verbose=False)
        result = results[0]
 
        bbox_frame = frame.copy()
        traj_frame = frame.copy()
 
        roi_poly = np.array(ROI_POINTS, dtype=np.int32)
        cv2.polylines(bbox_frame, [roi_poly], True, (0, 255, 255), 2)
        cv2.polylines(traj_frame, [roi_poly], True, (0, 255, 255), 2)
 
        persons = 0
        active = 0
 
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            persons = len(ids)
 
            for box, tid, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                last_seen_at[tid] = now
 
                if tid not in trajectories:
                    trajectories[tid] = deque(maxlen=TRAIL_MAXLEN)
                trajectories[tid].append((cx, cy))
                behaviors[tid] = classify_behavior(trajectories[tid])
                trail_color = TRAIL_COLORS[tid % len(TRAIL_COLORS)]
 
                update_heatmap(frame, cx, cy)
 
                inside = point_in_polygon((cx, cy), ROI_POINTS)
                if inside:
                    if tid not in entered_at:
                        entered_at[tid] = now
                    dwell = now - entered_at[tid]
                    if dwell >= LOITER_SECONDS:
                        alerted_ids.add(tid)
                        active += 1
                        color = (0, 0, 255)
                        label = f"ID {tid} LOITER {dwell:.1f}s"
                        with alert_log_lock:
                            exists = any(a["id"] == tid and a["status"] == "loitering"
                                         for a in list(alert_log)[-5:])
                            if not exists:
                                hourly_counts[datetime.now().hour] += 1
                                system_status["total_today"] += 1
                                alert_log.appendleft({
                                    "id": tid,
                                    "status": "loitering",
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "duration": round(dwell, 1),
                                    "behavior": behaviors.get(tid, "unknown"),
                                })
                                # 停留警報觸發 → 點亮 LED
                                threading.Thread(target=set_led, args=("on",), daemon=True).start()
                    else:
                        color = (0, 255, 0)
                        label = f"ID {tid} {dwell:.1f}s"
                else:
                    if tid in alerted_ids:
                        with alert_log_lock:
                            alert_log.appendleft({
                                "id": tid,
                                "status": "cleared",
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "duration": round(now - entered_at.get(tid, now), 1),
                                "behavior": behaviors.get(tid, "unknown"),
                            })
                    entered_at.pop(tid, None)
                    alerted_ids.discard(tid)
                    # 沒有任何人在停留 → 關掉 LED
                    color = (255, 255, 0)
                    label = f"ID {tid} outside"
 
                cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(bbox_frame, (cx, cy), 4, color, -1)
                cv2.putText(bbox_frame, f"{label} {conf:.2f}",
                            (x1, max(y1-8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
                if tid in trajectories and len(trajectories[tid]) > 1:
                    pts = list(trajectories[tid])
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)
                        c = (0, 0, 255) if tid in alerted_ids else trail_color
                        blended = tuple(int(ch * alpha) for ch in c)
                        cv2.line(traj_frame, pts[i-1], pts[i], blended, max(1, int(alpha*3)))
                cv2.circle(traj_frame, (cx, cy), 5, color, -1)
                beh = behaviors.get(tid, "")
                cv2.putText(traj_frame, f"ID {tid} [{beh}]",
                            (cx+8, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
 
        stale = [t for t, ls in last_seen_at.items() if now - ls > 2.0]
        for t in stale:
            last_seen_at.pop(t, None)
            entered_at.pop(t, None)
            alerted_ids.discard(t)
            trajectories.pop(t, None)
            behaviors.pop(t, None)

        system_status["active_alerts"] = active

        # active 是這幀實際偵測到的停留人數
        # 如果這幀沒有任何人停留超過 5 秒就關燈
        if active == 0:
            threading.Thread(target=set_led, args=("off",), daemon=True).start()
        system_status["persons_detected"] = persons
 
        fps_counter["count"] += 1
        if now - fps_counter["last_time"] >= 2.0:
            system_status["fps"] = round(fps_counter["count"] / (now - fps_counter["last_time"]), 1)
            fps_counter["count"] = 0
            fps_counter["last_time"] = now
 
        heat_frame = render_heatmap(frame)
        cv2.polylines(heat_frame, [roi_poly], True, (0, 255, 255), 2)
 
        q = [cv2.IMWRITE_JPEG_QUALITY, 75]
        _, buf_bbox = cv2.imencode(".jpg", bbox_frame, q)
        _, buf_traj = cv2.imencode(".jpg", traj_frame, q)
        _, buf_heat = cv2.imencode(".jpg", heat_frame, q)
 
        with frame_lock:
            latest_frame_bbox = buf_bbox.tobytes()
            latest_frame_traj = buf_traj.tobytes()
            latest_frame_heat = buf_heat.tobytes()
 
        time.sleep(0.05)
 
def gen_frames(which):
    while True:
        with frame_lock:
            if which == "bbox":
                frame = latest_frame_bbox
            elif which == "traj":
                frame = latest_frame_traj
            else:
                frame = latest_frame_heat
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.08)
 
@app.route("/")
def index():
    return render_template("index.html")
 
@app.route("/video_feed_bbox")
def video_feed_bbox():
    return Response(gen_frames("bbox"), mimetype="multipart/x-mixed-replace; boundary=frame")
 
@app.route("/video_feed_traj")
def video_feed_traj():
    return Response(gen_frames("traj"), mimetype="multipart/x-mixed-replace; boundary=frame")
 
@app.route("/video_feed_heat")
def video_feed_heat():
    return Response(gen_frames("heat"), mimetype="multipart/x-mixed-replace; boundary=frame")
 
@app.route("/api/status")
def api_status():
    return jsonify(system_status)
 
@app.route("/api/alerts")
def api_alerts():
    with alert_log_lock:
        return jsonify(list(alert_log)[:20])
 
@app.route("/api/hourly")
def api_hourly():
    return jsonify(hourly_counts)
 
if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)