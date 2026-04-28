#!/opt/conda/envs/yolov8_rtx5090/bin/python3.10
# -*- coding: utf-8 -*-
"""
多相機統合偵測節點（ROS1 Noetic）。

訂閱 5 個時間同步相機 topic，執行：
  - 橋樑 Stage1：對全部 5 路做批次推論
  - AC 路面 Stage1：僅對 ch2 / ch5（cam_id 1 / 4）執行
  - 各路 Stage2 分類（視 channel 而定）：
      cam_id 0、2（ch1、ch3）：伸縮縫高低差分類器（裁切 bbox）
      cam_id 3（ch4）        ：伸縮縫阻塞分類器（整張影像）
發布各路偵測結果，topic 格式與下游 GPS 節點合約完全一致。
"""
import os
import copy
from concurrent.futures import ThreadPoolExecutor

import rospy
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov8_ros_box_image_msgs.msg import BoundingBoxesWithImage

# ── 相機 / topic 對應 ──────────────────────────────────────────────────
NUM_CAMERAS = 5

TOPIC_NAMES = {
    0: 'right_120',
    1: 'front_120',
    2: 'left_120',
    3: 'back_120_1',
    4: 'back_120_2',
}

DIRECTIONS = {
    0: 'right',
    1: 'front',
    2: 'left',
    3: 'back-down',
    4: 'back',
}

# ── 橋樑偵測常數 ────────────────────────────────────────────────────────
BRIDGE_DESIRED_CLASS_IDS = {0, 1, 4, 6, 7, 8, 9}   # 異常類別（紅色）
BRIDGE_GREEN_CLASSES     = {2, 3, 5}                 # 正常類別（綠色）
BRIDGE_COLOR_MAP = {
    0: (0, 0, 255), 1: (0, 0, 255), 2: (0, 255, 0), 3: (0, 255, 0),
    4: (0, 0, 255), 5: (0, 255, 0), 6: (0, 0, 255), 7: (0, 0, 255),
    8: (0, 0, 255), 9: (0, 0, 255),
}

# ── Stage2 / 各路規則 ────────────────────────────────────────────────────
STAGE2_TRIGGER_CLS = {2, 6}
HEIGHT_CHANNELS    = {0, 2}     # ch1、ch3：對裁切 bbox 跑高低差分類器
GAP_CHANNELS       = {3}        # ch4：對整張影像跑阻塞分類器
ROAD_CHANNELS      = {1, 4}     # ch2、ch5：路面損壞模型
CH4_DISPLAY_CLS    = {6}        # 後下方相機只顯示 cls 6（伸縮縫阻塞）

# ── 路面損壞常數 ─────────────────────────────────────────────────────────
ROAD_DESIRED_CLASS_IDS = {0, 1, 2, 3}   # 四類全部視為異常
ROAD_COLOR             = (0, 165, 255)  # orange BGR，與原始專案一致


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main():
    rospy.init_node('multi_camera_detection_node', anonymous=False)

    # ── 讀取 ROS 參數 ────────────────────────────────────────────────────
    bridge_pt  = rospy.get_param('~bridge_pt_weight')
    bridge_rt  = rospy.get_param('~bridge_rt_weight')
    road_pt    = rospy.get_param('~road_pt_weight')
    road_rt    = rospy.get_param('~road_rt_weight')
    height_pt  = rospy.get_param('~height_pt_weight')
    height_rt  = rospy.get_param('~height_rt_weight')
    gap_pt     = rospy.get_param('~gap_pt_weight')
    gap_rt     = rospy.get_param('~gap_rt_weight')

    use_tensorrt = str(rospy.get_param('~tensorrt', 'False')).lower() in ('true', '1', 'yes')
    device       = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
    bridge_conf  = float(rospy.get_param('~bridge_conf', 0.3))
    road_conf    = float(rospy.get_param('~road_conf',   0.3))
    height_conf  = float(rospy.get_param('~height_conf', 0.3))
    gap_conf     = float(rospy.get_param('~gap_conf',    0.1))
    sync_slop    = float(rospy.get_param('~sync_slop',   0.05))
    road_y_ratio = float(rospy.get_param('~road_filter_y_ratio', 0.5))
    cam_frame    = rospy.get_param('~camera_frame', 'camera_frame')

    camera_topics = [rospy.get_param(f'~camera_topic_{i}') for i in range(NUM_CAMERAS)]

    # ── 載入模型 ─────────────────────────────────────────────────────────
    def _load_det(rt_path, pt_path):
        path = rt_path if use_tensorrt else pt_path
        if not os.path.exists(path):
            raise FileNotFoundError(f'找不到權重檔：{path}')
        if use_tensorrt:
            return YOLO(path)
        m = YOLO(pt_path).to(device)
        m.fuse()
        return m

    def _load_cls(rt_path, pt_path):
        path = rt_path if use_tensorrt else pt_path
        if not os.path.exists(path):
            raise FileNotFoundError(f'找不到權重檔：{path}')
        if use_tensorrt:
            return YOLO(path, task='classify')
        m = YOLO(pt_path).to(device)
        m.fuse()
        return m

    rospy.loginfo('正在載入模型...')
    bridge_model = _load_det(bridge_rt, bridge_pt)
    road_model   = _load_det(road_rt,   road_pt)
    height_model = _load_cls(height_rt, height_pt)
    gap_model    = _load_det(gap_rt,    gap_pt)

    # 暖機：各模型跑一次 dummy 避免第一幀延遲
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for m in (bridge_model, road_model, height_model, gap_model):
        try:
            m(dummy, device=device, verbose=False)
        except Exception:
            pass
    rospy.loginfo('所有模型載入並暖機完成。')

    # ── 建立 Publisher ───────────────────────────────────────────────────
    bridge_pubs = {}
    road_pubs   = {}
    for cam_id in range(NUM_CAMERAS):
        tn = TOPIC_NAMES[cam_id]
        bridge_pubs[cam_id] = {
            'bbox':   rospy.Publisher(f'/yolov8/{tn}/BoundingBoxes',           BoundingBoxes,         queue_size=1),
            'image':  rospy.Publisher(f'/{tn}/detection_image',                Image,                 queue_size=1),
            'filter': rospy.Publisher(f'/{tn}/filter_image_bbox',              BoundingBoxesWithImage, queue_size=1),
        }
        if cam_id in ROAD_CHANNELS:
            road_pubs[cam_id] = {
                'bbox':   rospy.Publisher(f'/yolov8/{tn}/BoundingBoxes_road_damage', BoundingBoxes,         queue_size=1),
                'image':  rospy.Publisher(f'/{tn}/detection_image_road_damage',      Image,                 queue_size=1),
                'filter': rospy.Publisher(f'/{tn}/filter_image_bbox_road_damage',    BoundingBoxesWithImage, queue_size=1),
            }

    cvbridge = CvBridge()
    executor = ThreadPoolExecutor(max_workers=16)

    # ── 時間同步 callback ────────────────────────────────────────────────
    def synced_callback(*image_msgs):
        try:
            frames = [cvbridge.imgmsg_to_cv2(m, desired_encoding='bgr8') for m in image_msgs]
        except CvBridgeError as e:
            rospy.logerr(f'cv_bridge 錯誤：{e}')
            return

        # 橋樑 Stage1：5 張批次推論
        try:
            bridge_results = bridge_model(frames, conf=bridge_conf, device=device, verbose=False)
        except Exception as e:
            rospy.logerr(f'橋樑推論錯誤：{e}')
            return
        
        # ★ 新增：提前算 ch4 gap_cls 整體結果，供 ch1/ch3 顯示
        gap_label = None
        try:
            sr = gap_model(frames[3], device=device, verbose=False)
            probs = sr[0].probs
            if probs is not None:
                top1      = int(probs.top1)
                top1conf  = float(probs.top1conf)
                top1name  = sr[0].names[top1]
                if top1conf >= gap_conf:
                    gap_label = top1name   # e.g. 'abnormal_expansion_joint' 或 'normal'
        except Exception as e:
            rospy.logerr(f'gap 預算錯誤：{e}')

        # 路面 Stage1：2 張批次推論（cam_id 1 與 4）
        road_result_map = {}
        try:
            road_batch = road_model(
                [frames[1], frames[4]], conf=road_conf, device=device, verbose=False
            )
            road_result_map = {1: road_batch[0], 4: road_batch[1]}
        except Exception as e:
            rospy.logerr(f'路面推論錯誤：{e}')

        for cam_id in range(NUM_CAMERAS):
            # 各路橋樑後處理丟入執行緒池（避免阻塞下一幀）
            # ch2/ch5 把 road_result 一起傳入，在同一張 detection_image 疊加
            road_res = copy.deepcopy(road_result_map[cam_id]) if cam_id in ROAD_CHANNELS and cam_id in road_result_map else None
            executor.submit(
                process_bridge_camera,
                cam_id, frames[cam_id],
                copy.deepcopy(bridge_results[cam_id]),  # deepcopy 防止 cls 改寫互相污染
                image_msgs[cam_id],
                bridge_pubs[cam_id],
                height_model, gap_model,
                height_conf, gap_conf,
                device, cam_frame,
                gap_label if cam_id in HEIGHT_CHANNELS else None,
                road_res, road_y_ratio,
            )
            if cam_id in ROAD_CHANNELS and cam_id in road_result_map:
                executor.submit(
                    process_road_camera,
                    cam_id, frames[cam_id],
                    copy.deepcopy(road_result_map[cam_id]),
                    image_msgs[cam_id],
                    road_pubs[cam_id],
                    road_y_ratio, cam_frame,
                    copy.deepcopy(bridge_results[cam_id]),
                )

    # ── 建立 Subscriber 與時間同步器 ────────────────────────────────────
    subs = [message_filters.Subscriber(t, Image) for t in camera_topics]
    ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=sync_slop)
    ts.registerCallback(synced_callback)

    rospy.loginfo(f'已訂閱：{camera_topics}（slop={sync_slop}s）')
    rospy.spin()


# ═══════════════════════════════════════════════════════════════════════
# 各路相機後處理
# ═══════════════════════════════════════════════════════════════════════

def process_bridge_camera(cam_id, frame, result, image_msg,
                           pubs, height_model, gap_model,
                           height_conf, gap_conf, device, cam_frame,
                           gap_label=None, road_result=None, road_y_ratio=0.5):
    # ── Stage2：依 channel 條件改寫 cls（例外不影響後續發布）──────────
    if cam_id in HEIGHT_CHANNELS or cam_id in GAP_CHANNELS:
        try:
            boxes   = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for i, cls in enumerate(classes):
                if int(cls) not in STAGE2_TRIGGER_CLS:
                    continue
                if cam_id in HEIGHT_CHANNELS:
                    # 裁切偵測到的 bbox 後跑高低差分類器
                    x1, y1, x2, y2 = boxes[i]
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        rospy.logwarn(f'cam{cam_id}: bbox[{i}] crop size=0，跳過')
                        continue
                    sr = height_model(crop, device=device, verbose=False)
                    probs = sr[0].probs
                    if probs is None:
                        rospy.logwarn(f'cam{cam_id}: height_model probs=None，跳過')
                        continue
                    top1     = int(probs.top1)
                    top1conf = float(probs.top1conf)
                    top1name = sr[0].names[top1]
                    rospy.logdebug(
                        f'cam{cam_id}({DIRECTIONS[cam_id]}) bbox[{i}] cls={int(cls)} '
                        f'→ height Stage2: {top1name} ({top1conf:.2f})')
                    if top1conf >= height_conf and top1 == 0:   # 0 = height_difference
                        result.boxes.cls[i] = 6
                        rospy.loginfo(
                            f'cam{cam_id}({DIRECTIONS[cam_id]})：高低差確認 '
                            f'({top1name} {top1conf:.2f}) → cls 改寫為 6')
                    elif top1conf < height_conf:
                        rospy.logdebug(
                            f'cam{cam_id}: Stage2 信心值 {top1conf:.2f} < 閾值 {height_conf}，不改寫')
                elif cam_id in GAP_CHANNELS:
                    # 對整張影像跑阻塞分類器
                    sr = gap_model(frame, device=device, verbose=False)
                    probs = sr[0].probs
                    if probs is None:
                        rospy.logwarn(f'cam{cam_id}: gap_model probs=None，跳過')
                        continue
                    top1     = int(probs.top1)
                    top1conf = float(probs.top1conf)
                    top1name = sr[0].names[top1]
                    rospy.logdebug(
                        f'cam{cam_id}({DIRECTIONS[cam_id]}) bbox[{i}] cls={int(cls)} '
                        f'→ gap Stage2: {top1name} ({top1conf:.2f})')
                    if top1conf >= gap_conf and top1 == 0:   # 0 = abnormal_expansion_joint
                        result.boxes.cls[i] = 6
                        rospy.loginfo(
                            f'cam{cam_id}({DIRECTIONS[cam_id]})：伸縮縫阻塞確認 '
                            f'({top1name} {top1conf:.2f}) → cls 改寫為 6')
                    elif top1conf < gap_conf:
                        rospy.logdebug(
                            f'cam{cam_id}: Stage2 信心值 {top1conf:.2f} < 閾值 {gap_conf}，不改寫')
        except Exception as e:
            rospy.logerr(f'process_bridge_camera Stage2 cam_id={cam_id}：{e}')

    # ── 發布（與 Stage2 例外隔離，確保一定會發布）────────────────────
    try:
        h, w  = frame.shape[:2]
        stamp = image_msg.header.stamp

        _pub_bridge_bbox_image(cam_id, result, h, w, stamp, pubs, cam_frame, frame,
                               gap_label=gap_label, road_result=road_result, road_y_ratio=road_y_ratio)
        _pub_bridge_filtered(cam_id, result, h, w, stamp, pubs, cam_frame, frame)

    except Exception as e:
        rospy.logerr(f'process_bridge_camera publish cam_id={cam_id}：{e}')


def process_road_camera(cam_id, frame, result, image_msg,
                         pubs, y_ratio, cam_frame, bridge_result=None):
    try:
        h, w  = frame.shape[:2]
        stamp = image_msg.header.stamp

        _pub_road_bbox_image(cam_id, result, frame, h, w, stamp, pubs, cam_frame, y_ratio,
                             bridge_result=bridge_result)
        _pub_road_filtered(cam_id, result, frame, h, w, stamp, pubs, cam_frame, y_ratio)

    except Exception as e:
        rospy.logerr(f'process_road_camera cam_id={cam_id}：{e}')


# ═══════════════════════════════════════════════════════════════════════
# 橋樑發布輔助函式
# ═══════════════════════════════════════════════════════════════════════

def _bridge_show_ok(cam_id, cls_id):
    """判斷此 cls 是否應顯示在完整偵測影像上。"""
    if cam_id == 3:
        return cls_id in CH4_DISPLAY_CLS   # ch4 只顯示 cls 6
    return True


def _bridge_anomaly(cam_id, cls_id):
    """判斷此 cls 是否屬於該相機的異常類別。"""
    if cam_id == 3:
        return cls_id in CH4_DISPLAY_CLS
    return cls_id in BRIDGE_DESIRED_CLASS_IDS


def _pub_bridge_bbox_image(cam_id, result, h, w, stamp, pubs, cam_frame, frame,
                           gap_label=None, road_result=None, road_y_ratio=0.5):
    """發布橋樑 BoundingBoxes（全部可顯示的框）與標註影像。
    ch2/ch5：road_result 不為 None 時，將路面損壞框疊加到橋樑影像上（同原始專案行為）。"""
    bb_msg = BoundingBoxes()
    bb_msg.header.stamp       = stamp
    bb_msg.image_header.stamp = stamp

    boxes   = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs   = result.boxes.conf.cpu().numpy()
    names   = result.names

    for box, cls, conf in zip(boxes, classes, confs):
        cls_id = int(cls)
        if not _bridge_show_ok(cam_id, cls_id):
            continue
        bb = BoundingBox()
        bb.xmin        = int(box[0])
        bb.ymin        = int(box[1])
        bb.xmax        = int(box[2])
        bb.ymax        = int(box[3])
        bb.Class       = names[cls_id]
        bb.probability = float(conf)
        bb_msg.bounding_boxes.append(bb)

    pubs['bbox'].publish(bb_msg)

    annotated = _draw_bridge(result, frame, cam_id, is_filtered=False)

    # ch2/ch5：路面損壞框疊加到橋樑影像上（與原始專案相同，橋樑先畫、路面疊在上方）
    if road_result is not None:
        annotated = _draw_road(road_result, annotated, h, road_y_ratio, is_filtered=False)

    # ch1/ch3 疊上 gap 分類結果文字
    if gap_label is not None:
        import cv2
        is_abnormal = 'abnormal' in gap_label.lower()
        color = (0, 80, 247) if is_abnormal else (133, 223, 2)   # 紅 / 綠
        text  = f'Gap: {gap_label}'
        cv2.putText(annotated, text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        
    pubs['image'].publish(_make_img_msg(annotated, h, w, stamp, cam_frame))


def _pub_bridge_filtered(cam_id, result, h, w, stamp, pubs, cam_frame, frame):
    """發布橋樑異常類別的 BoundingBoxesWithImage（filter_image_bbox）。"""
    classes = result.boxes.cls.cpu().numpy()
    names   = result.names

    found = {int(c) for c in classes if _bridge_anomaly(cam_id, int(c))}
    if not found:
        return   # 本幀無異常，不發布

    bb_msg = BoundingBoxes()
    bb_msg.header.stamp       = stamp
    bb_msg.image_header.stamp = stamp
    bb = BoundingBox()
    bb.Class = ', '.join(names[c] for c in found)
    bb_msg.bounding_boxes.append(bb)

    annotated = _draw_bridge(result, frame, cam_id, is_filtered=True)

    bwi = BoundingBoxesWithImage()
    bwi.header.stamp   = stamp
    bwi.bounding_boxes = bb_msg
    bwi.image          = _make_img_msg(annotated, h, w, stamp, cam_frame)
    pubs['filter'].publish(bwi)


def _draw_bridge(result, frame, cam_id, is_filtered):
    """繪製橋樑偵測框：綠色正常類別先畫，紅色異常類別後畫（疊在上方）。"""
    annotator = Annotator(frame.copy(), line_width=3, example=result.names)
    boxes   = result.boxes.xyxy
    classes = result.boxes.cls.tolist()
    names   = result.names

    for priority in ('green', 'red'):
        for box, cls in zip(boxes, classes):
            cls_id   = int(cls)
            is_green = cls_id in BRIDGE_GREEN_CLASSES
            if priority == 'green' and not is_green:
                continue
            if priority == 'red'   and     is_green:
                continue
            if is_filtered and not _bridge_anomaly(cam_id, cls_id):
                continue
            if not is_filtered and not _bridge_show_ok(cam_id, cls_id):
                continue
            color = BRIDGE_COLOR_MAP.get(cls_id, (255, 255, 255))
            annotator.box_label(box, label=names[cls_id], color=color)

    return annotator.result()


# ═══════════════════════════════════════════════════════════════════════
# 路面損壞發布輔助函式
# ═══════════════════════════════════════════════════════════════════════

def _pub_road_bbox_image(cam_id, result, frame, h, w, stamp, pubs, cam_frame, y_ratio,
                         bridge_result=None):
    """發布路面損壞 BoundingBoxes（套用 bbox 中心 y 過濾）與標註影像。
    bridge_result 不為 None 時，先畫橋樑 10 cls，再疊路面 4 cls（橙色）。"""
    bb_msg = BoundingBoxes()
    bb_msg.header.stamp       = stamp
    bb_msg.image_header.stamp = stamp

    boxes   = result.boxes.xyxy.cpu().numpy()
    xywh    = result.boxes.xywh.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confs   = result.boxes.conf.cpu().numpy()
    names   = result.names

    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
        if float(xywh[i][1]) > y_ratio * h:   # bbox 中心 y 偏低 → 過大誤檢，忽略
            continue
        bb = BoundingBox()
        bb.xmin        = int(box[0])
        bb.ymin        = int(box[1])
        bb.xmax        = int(box[2])
        bb.ymax        = int(box[3])
        bb.Class       = names[int(cls)]
        bb.probability = float(conf)
        bb_msg.bounding_boxes.append(bb)

    pubs['bbox'].publish(bb_msg)

    # 橋樑框先畫（紅/綠），路面框疊在上方（橙）
    base = _draw_bridge(bridge_result, frame, cam_id, is_filtered=False) if bridge_result is not None else frame
    annotated = _draw_road(result, base, h, y_ratio, is_filtered=False)
    pubs['image'].publish(_make_img_msg(annotated, h, w, stamp, cam_frame))


def _pub_road_filtered(cam_id, result, frame, h, w, stamp, pubs, cam_frame, y_ratio):
    """發布路面損壞異常類別的 BoundingBoxesWithImage（filter_image_bbox_road_damage）。"""
    classes = result.boxes.cls.cpu().numpy()
    xywh    = result.boxes.xywh.cpu().numpy()
    names   = result.names

    found = set()
    for i, cls in enumerate(classes):
        cls_id = int(cls)
        if cls_id not in ROAD_DESIRED_CLASS_IDS:
            continue
        if float(xywh[i][1]) > y_ratio * h:   # 同上過濾規則
            continue
        found.add(cls_id)

    if not found:
        return   # 本幀無路面損壞，不發布

    bb_msg = BoundingBoxes()
    bb_msg.header.stamp       = stamp
    bb_msg.image_header.stamp = stamp
    bb = BoundingBox()
    bb.Class = ', '.join(names[c] for c in found)
    bb_msg.bounding_boxes.append(bb)

    annotated = _draw_road(result, frame, h, y_ratio, is_filtered=True)

    bwi = BoundingBoxesWithImage()
    bwi.header.stamp   = stamp
    bwi.bounding_boxes = bb_msg
    bwi.image          = _make_img_msg(annotated, h, w, stamp, cam_frame)
    pubs['filter'].publish(bwi)


def _draw_road(result, frame, h, y_ratio, is_filtered):
    """繪製路面損壞偵測框，過濾掉 bbox 中心 y 超過閾值的框。"""
    annotator = Annotator(frame.copy(), line_width=3, example=result.names)
    boxes   = result.boxes.xyxy
    xywh    = result.boxes.xywh.cpu().numpy()
    classes = result.boxes.cls.tolist()
    names   = result.names

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        cls_id = int(cls)
        if float(xywh[i][1]) > y_ratio * h:
            continue
        if is_filtered and cls_id not in ROAD_DESIRED_CLASS_IDS:
            continue
        annotator.box_label(box, label=names[cls_id], color=ROAD_COLOR)

    return annotator.result()


# ═══════════════════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════════════════

def _make_img_msg(imgdata, height, width, stamp, cam_frame):
    """將 numpy 影像陣列封裝成 sensor_msgs/Image 訊息。"""
    msg              = Image()
    msg.header.stamp    = stamp
    msg.header.frame_id = cam_frame
    msg.height          = height
    msg.width           = width
    msg.encoding        = 'bgr8'
    msg.step            = width * 3
    msg.data            = np.array(imgdata).tobytes()
    return msg


if __name__ == '__main__':
    main()
