import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\jarvi\Desktop\人工智慧專題\runs\person_yolo_train\weights\best.pt")
cap = cv2.VideoCapture(r"C:\Users\jarvi\Desktop\人工智慧專題\test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.3, classes=[0], verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
