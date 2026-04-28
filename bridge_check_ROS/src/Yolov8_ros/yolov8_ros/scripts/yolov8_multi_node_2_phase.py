#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov8_ros_box_image_msgs.msg import BoundingBoxesWithImage
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from time import time, perf_counter
import os
from concurrent.futures import ThreadPoolExecutor
import copy


def main():
    # Initialize ROS node with a unique name based on the camera_id parameter
    rospy.init_node('yolov8_camera_node', anonymous=False)

    # Retrieve parameters
    camera_id = rospy.get_param('~camera_id', 0)    
    direction = rospy.get_param('~direction', 0)
    # --------------------------
    camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')
    pub_topic_bounding_boxes = rospy.get_param('~pub_topic_bounding_boxes', '/yolov8/bounding_boxes')
    pub_topic_detection_image = rospy.get_param('~pub_topic_detection_image', '/yolov8/detection_image')
    pub_topic_filtered_image_bbox = rospy.get_param('~pub_topic_filtered_image_bbox', '/yolov8/filtered_image_bbox')
    ptweight_path = rospy.get_param('~ptweight_path', '')
    rtweight_path = rospy.get_param('~rtweight_path', '')
    rospy.loginfo(f"after rtweight_path: {rtweight_path}")
    
    #exp_joint_height_rt_weight_path = rospy.get_param('~rtweight_path','') 
    #exp_joint_height_rt_weight_path = rospy.get_param('~expansion_joint_height_rt_weight_path','')    
    #rospy.loginfo(f"after exp_joint_height_rt_weight_path+++++: {exp_joint_height_rt_weight_path}")
    tensorrt_raw = rospy.get_param('~tensorrt', 'True')
    tensorrt = str(tensorrt_raw).lower() in ['true', '1', 'yes']
    device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
    camera_frame = rospy.get_param('~camera_frame', 'camera_frame')
    conf_threshold = rospy.get_param('~conf_threshold', 0.3)
   
    # Get the "desired_class_ids" parameter
    desired_class_ids_str = rospy.get_param('desired_class_ids', '[0, 1, 4, 6, 7, 8, 9]')  # Default provided for safety

    # Convert the string to a list of integers
    desired_class_ids = eval(desired_class_ids_str)  # Use eval cautiously; consider safer alternatives

    rospy.loginfo(f"Desired class IDs: {desired_class_ids}")
    
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    
    #exp_joint_height_rt_weight_path = rospy.get_param('~expansion_joint_height_rt_weight_path')
    #rospy.loginfo(f"after exp_joint_height_rt_weight_path: {exp_joint_height_rt_weight_path}")
    #exp_joint_gap_rt_weight_path = rospy.get_param('~expansion_joint_gap_rt_weight_path')
    #rospy.loginfo(f"after exp_joint_gap_rt_weight_path: {exp_joint_gap_rt_weight_path}")
    
    # Retrieve new weight file parameters for second-phase inference
    if tensorrt:
        #rospy.loginfo(f"before exp_joint_height_rt_weight_path: {exp_joint_height_rt_weight_path}")
        exp_joint_height_rt_weight_path = rospy.get_param('~expansion_joint_height_rt_weight_path')
        rospy.loginfo(f"after exp_joint_height_rt_weight_path: {exp_joint_height_rt_weight_path}")
        exp_joint_gap_rt_weight_path = rospy.get_param('~expansion_joint_gap_rt_weight_path')
        rospy.loginfo(f"after exp_joint_gap_rt_weight_path: {exp_joint_gap_rt_weight_path}")
    else:    
        exp_joint_height_pt_weight_path = rospy.get_param('~expansion_joint_height_pt_weight_path')        
        exp_joint_gap_pt_weight_path = rospy.get_param('~expansion_joint_gap_pt_weight_path')
        
    
    # Retrieve new confidence threshold parameters for second-phase inference    
    expansion_joint_height_conf = rospy.get_param('~expansion_joint_height_conf', 0.3)
    rospy.loginfo(f"expansion_joint_height_conf: {expansion_joint_height_conf}")    
    expansion_joint_gap_conf = rospy.get_param('~expansion_joint_gap_conf', 0.1)
    rospy.loginfo(f"expansion_joint_gap_conf: {expansion_joint_gap_conf}")

    # Initialize publishers
    position_pub = rospy.Publisher(pub_topic_bounding_boxes, BoundingBoxes, queue_size=1)
    image_pub = rospy.Publisher(pub_topic_detection_image, Image, queue_size=1)
    filtered_image_pub = rospy.Publisher(pub_topic_filtered_image_bbox, BoundingBoxesWithImage, queue_size=1)

    bridge = CvBridge()


    # Thread pool executor for parallel result processing
    executor = ThreadPoolExecutor(max_workers=2)

    # Load the model
    if tensorrt:
        if not os.path.exists(rtweight_path):
            rospy.logerr(f"Weight path {rtweight_path} does not exist.")
            return
        # For .engine
        model_first = YOLO(rtweight_path)
        print(f"Model loaded from {rtweight_path}")
    else:
        if not os.path.exists(ptweight_path):
            rospy.logerr(f"Please place the .pt weight as {ptweight_path}")
            return
        else:
            # For .pt
            model_first = YOLO(ptweight_path).to(device)
            model_first.fuse()
            print(f"Model loaded from {ptweight_path}")
            
     # Load the second-phase model based on camera_id (only for 0, 2, or 3)
    model_second = None
    if tensorrt:
        if camera_id in [0, 2]:            
            if not os.path.exists(exp_joint_height_rt_weight_path):
                rospy.logerr(f"Weight path {exp_joint_height_rt_weight_path} does not exist.")
                return
            model_second = YOLO(exp_joint_height_rt_weight_path, task='classify')
            rospy.loginfo(f"Second phase (height) model loaded from {exp_joint_height_rt_weight_path}")
        elif camera_id == 3:
            if not os.path.exists(exp_joint_gap_rt_weight_path):
                rospy.logerr(f"Weight path {exp_joint_gap_rt_weight_path} does not exist.")
                return
            model_second = YOLO(exp_joint_gap_rt_weight_path, task='classify')
            rospy.loginfo(f"Second phase (gap) model loaded from {exp_joint_gap_rt_weight_path}")
        # Warmup only if model_second is loaded (avoids NoneType error)
        if model_second is not None:
            rospy.loginfo(f"camera ID {camera_id} ({direction}): Warming up second-phase classification model")
            _ = model_second(dummy, device=device, verbose=False)  # No 'show' for classify  
        else:
            rospy.loginfo(f"camera ID {camera_id} ({direction}):  model_second is None")      
    else:
        if camera_id in [0, 2]:
            if not os.path.exists(exp_joint_height_pt_weight_path):
                rospy.logerr(f"Weight path {exp_joint_height_pt_weight_path} does not exist.")
                return
            model_second = YOLO(exp_joint_height_pt_weight_path).to(device)
            model_second.fuse()
            rospy.loginfo(f"Second phase (height) model loaded from {exp_joint_height_pt_weight_path}")
        elif camera_id == 3:
            if not os.path.exists(exp_joint_gap_pt_weight_path):
                rospy.logerr(f"Weight path {exp_joint_gap_pt_weight_path} does not exist.")
                return
            model_second = YOLO(exp_joint_gap_pt_weight_path).to(device)
            model_second.fuse()
            rospy.loginfo(f"Second phase (gap) model loaded from {exp_joint_gap_pt_weight_path}")       

    # Define the image callback
    def image_callback(image_msg):
        # Reconstruct image from ROS Image message
        try:
            color_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        start = perf_counter()
        results = model_first(color_image, show=False, conf=conf_threshold, device=device, verbose=False)
        end = perf_counter()
        delay = 1000*(end - start)
        #if camera_id == 0:
        #    print(f"inference delay: {delay:.2f}ms")

        # Process and publish results
        start = perf_counter()
        # publish_results(image_msg, color_image, results)
        ##executor.submit(publish_results, image_msg, color_image, results)
        results_writeable = copy.deepcopy(results)
        executor.submit(publish_results, image_msg, color_image, results_writeable)
        end = perf_counter()
        delay = 1000*(end - start)
        #if camera_id == 0:
        #    print(f"result processing delay: {delay:.2f}ms")


    def publish_results(image_msg, color_image, results):
         # ----- Second-phase inference update -----
        if camera_id in [0, 2, 3] and model_second is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (N, 4)
            classes = results[0].boxes.cls.cpu().numpy()   # shape: (N,)
            for i, cls in enumerate(classes):
                if int(cls) in [2, 6]:
                    # For camera_id 0 or 2, crop the detected bounding box and run second-phase inference
                    if camera_id in [0, 2]:                        
                        x1, y1, x2, y2 = boxes[i]
                        crop_img = color_image[int(y1):int(y2), int(x1):int(x2)]
                        # rospy.loginfo(f"Camera {camera_id}: class 2 or 6 before model_second ++++++).")
                        second_results = model_second(crop_img, device=device, verbose=False)
                        # rospy.loginfo(f"Camera {camera_id}: class 2 or 6 after model_second ++++++).")
                        second_cls = int(second_results[0].probs.top1)
                        conf = float(second_results[0].probs.top1conf)
                        # Mapping: {0: 'height_difference', 1: 'no_height_difference'}
                        if second_cls == 0:
                            rospy.loginfo(f"Camera {camera_id} ({direction}): YOLO-CLS indicates height difference; updating class to 6 ('block expansion joint').")
                            results[0].boxes.cls[i] = 6
                        #else:
                        #    rospy.loginfo(f"Camera {camera_id}: no_height_difference ")   

                    # For camera_id 3, run second-phase inference on the entire image
                    elif camera_id == 3:
                        second_results = model_second(color_image, device=device, verbose=False)
                        second_cls = int(second_results[0].probs.top1)
                        conf = float(second_results[0].probs.top1conf)
                        # Mapping: {0: 'abnormal_expansion_joint', 1: 'normal_expansion_joint'}
                        if second_cls == 0:
                            rospy.loginfo(f"Camera {camera_id} ({direction}): YOLO-CLS indicates abnormal expansion joint; updating class to 6 ('block expansion joint').")
                            results[0].boxes.cls[i] = 6

        # -------------------------------------------
        detect_show(results, image_msg.height, image_msg.width, image_msg.header.stamp, color_image)
        filtered_detect_show(results, image_msg.height, image_msg.width, image_msg.header.stamp, color_image)


    def detect_show(results, height, width, stamp, image):
        # Use local variables from main
        boundingBoxes = BoundingBoxes()
        boundingBoxes.header = Header()
        boundingBoxes.header.stamp = stamp
        boundingBoxes.image_header = Header()
        boundingBoxes.image_header.stamp = stamp

        # Extract bounding box data using NumPy
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        classes = results[0].boxes.cls.cpu().numpy()  # class IDs
        confidences = results[0].boxes.conf.cpu().numpy()  # confidence scores
        names = results[0].names  # class names

        # Create BoundingBox objects using vectorized operations
        for box, cls, conf in zip(boxes, classes, confidences):
            boundingBox = BoundingBox()
            boundingBox.xmin = int(box[0])
            boundingBox.ymin = int(box[1])
            boundingBox.xmax = int(box[2])
            boundingBox.ymax = int(box[3])
            boundingBox.Class = names[int(cls)]
            boundingBox.probability = float(conf)
            boundingBoxes.bounding_boxes.append(boundingBox)

        # Publish bounding boxes
        position_pub.publish(boundingBoxes)

        # Use existing plot_bboxes function for visualization and image publishing
        frame, class_ids = plot_bboxes(results=results, im0=image, is_filtered=False)

        # Publish detection image
        publish_image(frame, height, width, stamp, image_pub, camera_frame)


    def filtered_detect_show(results, height, width, stamp, image):
        filted_count = 0
        BoundingBoxesWithImage_temp = BoundingBoxesWithImage()
        boundingBoxes = BoundingBoxes()
        boundingBoxes.header = Header()
        boundingBoxes.header.stamp = stamp
        boundingBoxes.image_header = Header()
        boundingBoxes.image_header.stamp = stamp
        
        

        # Initialize an empty set to hold found classes
        found_classes_set = set()

        for i in range(len(results[0].boxes.cls)):
            cls_id = int(results[0].boxes.cls[i])
            if cls_id in desired_class_ids:
                if camera_id == 3 and cls_id != 6: # camera_id 3(back down camera), it only allow cls_id 6 (block expansion joint)
                   continue
                   
                # 取得 bbox Y (假設 xywh 格式，index 1 為 height)
                bbox_y = results[0].boxes.xywh[i][1]

                # --- 針對 camera 1(front) 和 4(back_2) 的特定過濾邏輯 cls_id 4(abnormal drainage hole)---
                if (camera_id == 1 or camera_id == 4) and cls_id == 4:                              
                    # Debug: 印出高度與底部位置，方便您確認
                    print(f"Camera {camera_id} ({direction}): - bbox_y: {bbox_y}  Litmit : {height}/2") 

                    # 條件 1: 如果高度超過畫面一半 (過濾超大誤檢)
                    if bbox_y > (height / 2):
                        print("   -> Ignored: Too tall")
                        continue
                                  
                class_description = results[0].names[cls_id]
                found_classes_set.add(class_description)
                filted_count += 1
              

        if not filted_count:
            return  # No desired objects found

        # Join the found classes into a single string
        found_classes = ", ".join(found_classes_set)
        print(f"Camera {camera_id} ({direction}): - Found classes: {found_classes}")        
        

        boundingBox = BoundingBox()
        boundingBox.Class = found_classes
        boundingBoxes.bounding_boxes.append(boundingBox)

        BoundingBoxesWithImage_temp.bounding_boxes = boundingBoxes

        frame, class_ids = plot_bboxes(results, image, is_filtered=True)

        image_temp = Image()
        header = Header()
        header.stamp = stamp
        header.frame_id = camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(frame).tobytes()
        image_temp.header = header
        image_temp.step = width * 3

        BoundingBoxesWithImage_temp.image = image_temp
        BoundingBoxesWithImage_temp.header.stamp = stamp

        filtered_image_pub.publish(BoundingBoxesWithImage_temp)

    def plot_bboxes(results, im0, is_filtered=False):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        color_map = {
            0: (0, 80, 247),     # spalled concrete - 橘/紅
            1: (0, 80, 247),     # corroded bolt
            2: (133, 223, 2),    # expansion joint - 綠
            3: (133, 223, 2),    # normal drainage hole - 綠
            4: (0, 80, 247),     # abnormal drainage hole
            5: (133, 223, 2),    # normal bolt - 綠
            6: (0, 80, 247),     # block expansion joint
            7: (0, 80, 247),     # rust steel pipe
            8: (0, 80, 247),     # barrier damage
            9: (0, 80, 247),     # blocked drainage hole
        }

        green_classes = {2, 3, 5}
        class_ids = []
        annotator = Annotator(im0.copy(), line_width=3, example=results[0].names)
        boxes = results[0].boxes.xyxy
        clss = results[0].boxes.cls.tolist()
        names = results[0].names

        # 分類兩組：綠色類別先畫、紅色類別後畫
        for priority in ['green', 'red']:
            for box, cls in zip(boxes, clss):
                cls = int(cls)
                if (priority == 'green' and cls not in green_classes) or (priority == 'red' and cls in green_classes):
                    continue

                if is_filtered and cls not in desired_class_ids:
                    continue  # skip non-filtered class

                if camera_id == 3 and cls not in (2, 6): # camera_id 3(back down camera), it only allow cls_id 2(expansion joint) or 6 (block expansion joint)
                    continue

                class_ids.append(cls)
                label = names[cls]
                #print(f"cls: {cls}, label: {names[cls]}")                
                box_color = color_map.get(cls, (255, 255, 255))  # fallback: white
                annotator.box_label(box, label=label, color=box_color)

        return annotator.result(), class_ids


    def publish_image(imgdata, height, width, stamp, image_pub, camera_frame):
        image_temp = Image()
        header = Header()
        header.stamp = stamp
        header.frame_id = camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3

        image_pub.publish(image_temp)

    # Subscribe to the camera topic
    rospy.Subscriber(camera_topic, Image, image_callback, queue_size=1, buff_size=52428800)

    rospy.spin()

if __name__ == '__main__':
    main()
