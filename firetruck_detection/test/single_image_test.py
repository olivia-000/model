import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ultralytics import YOLO

from typing import Tuple, List

from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2

def classify_image(model, image=None):
    # 執行分類
    results = model(image, verbose=False)
    
    # 印出分類結果
    print(f"Image: {image}")
    print("Classification Results:")

    if len(results) > 0:
        # Get predictions from first result
        result = results[0]
        
        # Get class names and confidences
        probs = result.probs  # Classification results
        top5_list = probs.top5
        top5conf_list = probs.top5conf
    
        class_names = [result.names[i] for i in top5_list]
        class_names = [f"{class_names[i]}: {top5conf_list[i]:.2f}" for i in range(len(class_names))]

        print(f"Top 5 class names: {', '.join(class_names)}")
    else:
        print("No objects detected in the image.")
        
def _plot_bboxes(results: Results, im0: np.ndarray) -> Tuple[np.ndarray, float, List[int]]:
    """Plots bounding boxes on an image given detection results; returns annotated image, confidence value and class IDs."""
    class_ids = [] 

    annotated_frame = im0.copy()
    annotator = Annotator(annotated_frame, 10, results.names)

    boxes = results.boxes.xyxy.cpu()
    # Make the boxes top-left origin lower than y = 100
    boxes[:, 1] = boxes[:, 1].clamp(min=200)
    
    clss = results.boxes.cls.cpu().tolist()
    confs = results.boxes.conf.cpu().tolist()  # Assuming confidence scores are here
    names = results.names

    for box, cls, conf in zip(boxes, clss, confs):
        class_ids.append(cls)
        label = f"{names[int(cls)]}: {conf:.2f}"  # Format label to include confidence
        annotator.box_label(box, label=label, color=colors(int(cls), True))

    return annotated_frame, confs[0], class_ids
    
def object_detection(model, image_name=None):
    # 獲取當前目錄中第一個符合條件的圖片檔案
    image = cv2.imread(image_name)
    
    # 執行物件檢測
    results = model(image, verbose=False)
    frame_h = image.shape[0]
    boxes = results[0].boxes.xywh
    center_y = (boxes[:, 1] + boxes[:, 3] / 2).cpu().numpy()

    is_within_acceptance_range = np.any(center_y > frame_h * 1/3)
    print(f"Center of the bbox is below the horizontal line: {is_within_acceptance_range}")

    annotated_frame, _, _ = _plot_bboxes(results[0], image)
    # save annotated image
    annotated_image_file = f"annotated_{image}"
    cv2.imwrite(annotated_image_file, annotated_frame)
    
    # 印出檢測結果
    print(f"Image: {image}")
    print("Detection Results:")
    
    if len(results) > 0:
        # Get bbox coordinates and confidences
        bboxes = results[0].boxes.xyxy.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        
        # Print the bbox coordinates and confidences
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            conf = confs[i]
            print(f"Bounding box: {bbox} Confidence: {conf:.2f}")
    else:
        print("No objects detected in the image.")
    
    # Object cropping
    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0].cpu())
    x_offset, y_offset = 0, 0
    margin = 100 # Having a margin around the bounding box makes it easier to identify the object
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)
    cropped_frame = image[y1:y2, x1:x2]

    return cropped_frame

if __name__ == "__main__":
    # Load models once
    detection_model = YOLO("../weights/0924_v10l_run131.pt")
    classification_model = YOLO("../weights/yolo11x-cls.pt")

    input_image_path = "image_under_test.jpg"
    cropped_image_path = "cropped.jpg"
    
    cropped = object_detection(detection_model, input_image_path)

    # save cropped image
    cv2.imwrite(cropped_image_path, cropped)

    classify_image(classification_model, image=cropped)
