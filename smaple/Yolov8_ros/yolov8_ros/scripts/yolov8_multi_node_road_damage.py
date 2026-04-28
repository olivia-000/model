#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
from yolov8_ros_box_image_msgs.msg import BoundingBoxesWithImage
from cv_bridge import CvBridge
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from time import time, perf_counter
import os
from concurrent.futures import ThreadPoolExecutor


def main():
    # Initialize ROS node with a unique name based on the camera_id parameter
    rospy.init_node('yolov8_camera_node', anonymous=False)

    # Retrieve parameters
    camera_id = rospy.get_param('~camera_id', 0)
    camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')
    pub_topic_bounding_boxes = rospy.get_param('~pub_topic_bounding_boxes', '/yolov8/bounding_boxes')
    pub_topic_detection_image = rospy.get_param('~pub_topic_detection_image', '/yolov8/detection_image')
    pub_topic_filtered_image_bbox = rospy.get_param('~pub_topic_filtered_image_bbox', '/yolov8/filtered_image_bbox')
    ptweight_path = rospy.get_param('~ptweight_path', '')
    rtweight_path = rospy.get_param('~rtweight_path', '')
    tensorrt = rospy.get_param('~tensorrt', True)
    device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
    camera_frame = rospy.get_param('~camera_frame', 'camera_frame')
    conf_threshold = rospy.get_param('~conf_threshold', 0.3)
   
    # Get the "desired_class_ids" parameter
    desired_class_ids_str = rospy.get_param('desired_class_ids', '[0, 1, 2, 3]')  # Default provided for safety

    # Convert the string to a list of integers
    desired_class_ids = eval(desired_class_ids_str)  # Use eval cautiously; consider safer alternatives

    rospy.loginfo(f"Desired class IDs: {desired_class_ids}")
  

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
        model = YOLO(rtweight_path)
        print(f"Model loaded from {rtweight_path}")
    else:
        if not os.path.exists(ptweight_path):
            rospy.logerr(f"Please place the .pt weight as {ptweight_path}")
            return
        else:
            # For .pt
            model = YOLO(ptweight_path).to(device)
            model.fuse()
            print(f"Model loaded from {ptweight_path}")

    # Define the image callback
    def image_callback(image_msg):
        # Reconstruct image from ROS Image message
        try:
            color_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridge.Error as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        start = perf_counter()
        results = model(color_image, show=False, conf=conf_threshold, device=device, verbose=False)
        end = perf_counter()
        delay = 1000*(end - start)
        #if camera_id == 0:
        #    print(f"inference delay: {delay:.2f}ms")

        # Process and publish results
        start = perf_counter()
        # publish_results(image_msg, color_image, results)
        executor.submit(publish_results, image_msg, color_image, results)
        end = perf_counter()
        delay = 1000*(end - start)
        #if camera_id == 0:
        #    print(f"result processing delay: {delay:.2f}ms")


    def publish_results(image_msg, color_image, results):
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
                class_description = results[0].names[cls_id]
                found_classes_set.add(class_description)
                filted_count += 1
              

        if not filted_count:
            return  # No desired objects found

        # Join the found classes into a single string
        found_classes = ", ".join(found_classes_set)
        print(f"Camera {camera_id} - Found classes: {found_classes}")

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
        class_ids = []
        annotator = Annotator(im0.copy(), 3, results[0].names)
        boxes = results[0].boxes.xyxy
        clss = results[0].boxes.cls.tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            if is_filtered:
                if cls in desired_class_ids:
                    class_ids.append(cls)                    
                    annotator.box_label(box, label=names[int(cls)], color=colors(int(6), True))  # 6 color is red
            else:
                class_ids.append(cls)                
                if cls in desired_class_ids:
                    annotator.box_label(box, label=names[int(cls)], color=colors(int(6), True))  # 6 color is red
                else:
                    annotator.box_label(box, label=names[int(cls)], color=colors(int(0), True))  # 0 color is green
                
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
