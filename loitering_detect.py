import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Loitering detection using YOLO + tracking")
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\jarvi\Desktop\人工智慧專題\runs\person_yolo_train\weights\best.pt",
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=r"C:\Users\jarvi\Desktop\人工智慧專題\test.mp4",
        help="Video path, webcam index like 0, RTSP URL, or image folder",
    )
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker config")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--loiter-seconds", type=float, default=10.0, help="Alert threshold in seconds")
    parser.add_argument("--grace-seconds", type=float, default=1.0, help="Tolerance for temporary missed detections")
    parser.add_argument("--show", action="store_true", help="Show live window")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument(
        "--output",
        type=str,
        default=r"C:\Users\jarvi\Desktop\人工智慧專題\loitering_output.mp4",
        help="Saved video path",
    )
    return parser.parse_args()


def parse_source(src: str):
    return int(src) if src.isdigit() else src


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_roi(frame, polygon):
    cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(
        frame,
        "ROI",
        tuple(polygon[0]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_writer(cap, output_path, fallback_shape=None):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap is not None else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap is not None else 0

    if fps <= 0:
        fps = 20.0

    if width <= 0 or height <= 0:
        if fallback_shape is None:
            raise ValueError("Cannot determine output video size.")
        height, width = fallback_shape[:2]

    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    source = parse_source(args.source)
    model = YOLO(str(model_path))

    # ===== ROI 設定區 =====
    # 直接改這 4 個點即可
    roi_points = [
        (0, 0),
        (1280, 0),
        (1280, 960),
        (0, 960),
    ]
    roi_polygon = np.array(roi_points, dtype=np.int32)

    entered_at = {}
    last_seen_at = {}
    alerted_ids = set()

    writer = None
    if str(source).startswith("http"):
        from esp32_source import ESP32Stream
        cap = ESP32Stream(source)
    else:
        cap = cv2.VideoCapture(source)

    # 等第一幀
    print("等待相機畫面...")
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.1)
    print("相機連線成功！")

    if args.save:
        writer = make_writer(None, args.output, fallback_shape=frame.shape)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if str(source).startswith("http"):
                time.sleep(0.05)
            continue

        frame = frame.copy()
        now = time.time()

        results = model.track(
            frame,
            conf=args.conf,
            iou=args.iou,
            tracker=args.tracker,
            persist=True,
            classes=[0],
            verbose=False,
        )
        result = results[0]

        draw_roi(frame, roi_polygon)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            clss = result.boxes.cls.int().cpu().tolist()

            for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                last_seen_at[track_id] = now
                inside_roi = point_in_polygon((cx, cy), roi_polygon)

                if inside_roi:
                    if track_id not in entered_at:
                        entered_at[track_id] = now
                    dwell = now - entered_at[track_id]
                    if dwell >= args.loiter_seconds:
                        alerted_ids.add(track_id)
                        color = (0, 0, 255)
                        label = f"ID {track_id} LOITER {dwell:.1f}s"
                    else:
                        color = (0, 255, 0)
                        label = f"ID {track_id} {dwell:.1f}s"
                else:
                    entered_at.pop(track_id, None)
                    alerted_ids.discard(track_id)
                    color = (255, 255, 0)
                    label = f"ID {track_id} outside"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                cv2.putText(
                    frame,
                    f"{label} conf={conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        stale_ids = []
        for track_id, last_seen in last_seen_at.items():
            if now - last_seen > args.grace_seconds:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            last_seen_at.pop(track_id, None)
            entered_at.pop(track_id, None)
            alerted_ids.discard(track_id)

        cv2.putText(
            frame,
            f"Loiter Threshold: {args.loiter_seconds:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Current Alerts: {len(alerted_ids)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if alerted_ids else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if args.show:
            cv2.imshow("Loitering Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if args.save:
            if writer is None:
                writer = make_writer(None, args.output, fallback_shape=frame.shape)
            writer.write(frame)

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Done. Output: {args.output}" if args.save else "Done.")


if __name__ == "__main__":
    main()
