import datetime
import cv2
from ultralytics import YOLO
import math 

# object classes
coco128 = open('./yolov8_pretrained/coco128.txt', 'r')
data = coco128.read()
classNames = data.split('\n')
coco128.close()

# model
model = YOLO("yolo-Weights/yolov8n.pt")

video_path = 0
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error video file")

while True:
    start = datetime.datetime.now()
    ret, frame = cap.read()
    detection = model(frame, stream=True)

    # coordinates
    for obj in detection:
        boxes = obj.boxes

        for box in boxes: #[xmin, ymin, xmax, ymax, confidence_score, class_id]
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

    end = datetime.datetime.now()
    total = (end - start).total_seconds()

    fps = f'FPS: {1 / total:.2f}'
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()